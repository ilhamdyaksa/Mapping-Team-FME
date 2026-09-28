import io
import math
import re
import datetime
import folium
from folium import plugins
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Permanen RedZone 3 Kabupaten
REDZONE_KABUPATEN = ["puncak jaya", "pegunungan bintang", "tolikara"]

# ==============================================================================
# 1. HELPER & UTILITY FUNCTIONS
# ==============================================================================

def temukan_kolom(df, keywords):
    if df is None or df.empty:
        return None
    
    cols_clean = {col: re.sub(r'[^a-zA-Z0-9]', '', str(col).strip().lower()) for col in df.columns}
    keywords_clean = [re.sub(r'[^a-zA-Z0-9]', '', str(kw).strip().lower()) for kw in keywords]
    
    for col, c_clean in cols_clean.items():
        if c_clean in keywords_clean:
            return col
            
    for col, c_clean in cols_clean.items():
        if any(kw in c_clean for kw in keywords_clean):
            return col
            
    return None

def clean_to_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace('%', '', regex=False)
        .str.replace(',', '.', regex=False)
        .str.strip(),
        errors='coerce'
    ).fillna(0)

def hitung_jarak_haversine_vec(lat1, lon1, lat2_series, lon2_series):
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2_series.to_numpy()), np.radians(lon2_series.to_numpy())

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

@st.cache_data(show_spinner=False)
def get_osrm_route(lat1, lon1, lat2, lon2):
    try:
        url_pt1 = "http://router.project-osrm.org/route/v1/driving/"
        url_pt2 = f"{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
        response = requests.get(url_pt1 + url_pt2, timeout=3)
        if response.status_code == 200:
            res = response.json()
            if res.get("routes") and len(res["routes"]) > 0:
                distance_km = res["routes"][0]["distance"] / 1000.0
                duration_min = res["routes"][0]["duration"] / 60.0
                geometry = res["routes"][0]["geometry"]
                return True, distance_km, duration_min, geometry
    except Exception:
        pass
    return False, None, None, None

def analisa_moda_per_site(site_row, dist_straight_km, has_osrm_road):
    row_str = " ".join([str(v).lower() for v in site_row.values])
    
    if any(k in row_str for k in ['sungai', 'rawa', 'longboat', 'boat', 'sampan', 'dermaga']):
        return "Sungai/Rawa (Longboat)", "Tersedia jalur sungai (Akses Boat)", True
    
    if any(k in row_str for k in ['heli', 'pesawat', 'airstrip', 'perintis', 'terisolir']):
        return "Udara/Helikopter", "Area pegunungan (Penerbangan Charter)", False

    if has_osrm_road:
        return "Darat 4x4", "Akses jalan darat terhubung", True
    
    if dist_straight_km > 75.0:
        return "Udara/Helikopter", f"Sangat jauh ({dist_straight_km:.1f} KM) tanpa jalur darat", False
    elif dist_straight_km > 30.0:
        return "Darat Offroad / Perahu", "Jalur perintisan lokal (Motor Offroad / Boat)", True
    else:
        return "Darat 4x4 / Offroad", "Akses darat lokal terdekat", True

# ==============================================================================
# 2. LOGIKA EVALUASI PERFORMANCE DAILY & CLUSTER GROUPING
# ==============================================================================

def process_performance_aging_daily(df_perf, site_col, date_col, avail_col, t_start_b, t_end_b, t_end_l):
    df_perf[date_col] = pd.to_datetime(df_perf[date_col]).dt.date

    d_start_b = pd.to_datetime(t_start_b).date()
    d_end_b = pd.to_datetime(t_end_b).date()
    d_latest = pd.to_datetime(t_end_l).date()

    df_perf['is_down'] = df_perf[avail_col].apply(lambda x: 1 if x == 0 else 0)

    df_baseline = df_perf[(df_perf[date_col] >= d_start_b) & (df_perf[date_col] <= d_end_b)]
    site_baseline_down = df_baseline.groupby(site_col)['is_down'].mean()
    sites_baseline_1wk = set(site_baseline_down[site_baseline_down == 1.0].index)

    df_eval = df_perf[(df_perf[date_col] >= d_start_b) & (df_perf[date_col] <= d_latest)]
    df_pivot = df_eval.pivot_table(index=site_col, columns=date_col, values='is_down', aggfunc='first').fillna(0)
    
    sorted_dates = sorted(df_pivot.columns, reverse=True)
    
    sites_down_gt_2days_all = set()
    for site, row in df_pivot.iterrows():
        consecutive_down = 0
        for d in sorted_dates:
            if row[d] == 1:
                consecutive_down += 1
            else:
                break
        if consecutive_down > 2:
            sites_down_gt_2days_all.add(site)

    df_latest_snapshot = df_perf[df_perf[date_col] == d_latest]
    sites_up_latest = set(df_latest_snapshot[df_latest_snapshot['is_down'] == 0][site_col])
    sites_recovered = sites_baseline_1wk.intersection(sites_up_latest)

    sites_tambahan_2days = sites_down_gt_2days_all.difference(sites_baseline_1wk)
    active_baseline_down = sites_baseline_1wk.difference(sites_recovered)
    all_target_down_sites = active_baseline_down.union(sites_tambahan_2days)

    return (
        list(all_target_down_sites),
        list(sites_tambahan_2days),
        len(sites_baseline_1wk),
        len(sites_recovered),
        len(sites_tambahan_2days)
    )

def temukan_cluster_berdekatan(df_cluster_input, lat_col, lon_col, max_dist_km=30.0):
    df_work = df_cluster_input.copy().reset_index(drop=True)
    n = len(df_work)
    if n < 2:
        df_work['cluster_id'] = -1
        df_work['cluster_size'] = n
        return df_work

    visited = [False] * n
    cluster_labels = [-1] * n
    current_cluster_id = 1

    for i in range(n):
        if visited[i]:
            continue
        
        lat1, lon1 = df_work.loc[i, lat_col], df_work.loc[i, lon_col]
        dists = hitung_jarak_haversine_vec(lat1, lon1, df_work[lat_col], df_work[lon_col])
        neighbors = np.where(dists <= max_dist_km)[0].tolist()

        if len(neighbors) >= 2:
            for idx in neighbors:
                if not visited[idx]:
                    visited[idx] = True
                    cluster_labels[idx] = current_cluster_id
            current_cluster_id += 1

    df_work['cluster_id'] = cluster_labels
    cluster_counts = df_work['cluster_id'].value_counts().to_dict()
    df_work['cluster_size'] = df_work['cluster_id'].map(cluster_counts).fillna(1).astype(int)
    
    return df_work

def generate_all_am_export_data(df_assigned, col_am, col_kab, col_kec, col_site_name, col_lat, col_lon, use_osrm):
    all_rows = []
    
    for am_name, df_am in df_assigned.groupby(col_am):
        df_clustered = temukan_cluster_berdekatan(df_am, col_lat, col_lon, max_dist_km=30.0)
        
        df_multi = df_clustered[df_clustered['cluster_id'] != -1]
        if not df_multi.empty:
            for c_id, group in df_multi.groupby('cluster_id'):
                first_row = group.iloc[0]
                curr_lat, curr_lon = first_row[col_lat], first_row[col_lon]
                
                for idx_g, row_g in group.reset_index().iterrows():
                    lat_g, lon_g = row_g[col_lat], row_g[col_lon]
                    dist_km = hitung_jarak_haversine_vec(curr_lat, curr_lon, pd.Series([lat_g]), pd.Series([lon_g]))[0]
                    
                    if use_osrm:
                        has_road, _, _, _ = get_osrm_route(curr_lat, curr_lon, lat_g, lon_g)
                    else:
                        has_road = False

                    moda, ket_akses, _ = analisa_moda_per_site(row_g, dist_km, has_road)
                    s_name_val = row_g[col_site_name] if col_site_name and col_site_name in row_g else '-'
                    
                    all_rows.append({
                        'Area Manager PIC': am_name,
                        'Group Cluster ID': f"Cluster #{c_id}",
                        'Stop #': idx_g + 1,
                        'Site ID': row_g['site_id_clean'],
                        'Site Name': s_name_val,
                        'Kabupaten': row_g[col_kab],
                        'Kecamatan': row_g[col_kec] if col_kec else '-',
                        'Kategori Down': row_g['kategori_down'],
                        'Jarak Leg (KM)': round(dist_km, 2),
                        'Moda Akses': moda,
                        'Kriteria Akses Jalan': ket_akses,
                        'Status Assignment': 'Plan Visit',
                        'Remark Manual Tim': ''
                    })

        df_single = df_clustered[df_clustered['cluster_id'] == -1]
        if not df_single.empty:
            for idx_s, row_s in df_single.reset_index().iterrows():
                s_name_val = row_s[col_site_name] if col_site_name and col_site_name in row_s else '-'
                
                all_rows.append({
                    'Area Manager PIC': am_name,
                    'Group Cluster ID': 'Single Site',
                    'Stop #': idx_s + 1,
                    'Site ID': row_s['site_id_clean'],
                    'Site Name': s_name_val,
                    'Kabupaten': row_s[col_kab],
                    'Kecamatan': row_s[col_kec] if col_kec else '-',
                    'Kategori Down': row_s['kategori_down'],
                    'Jarak Leg (KM)': 0.0,
                    'Moda Akses': '-',
                    'Kriteria Akses Jalan': '-',
                    'Status Assignment': 'Plan Visit',
                    'Remark Manual Tim': ''
                })

    return pd.DataFrame(all_rows)

# ==============================================================================
# 3. STREAMLIT FRONTEND & MAIN APP
# ==============================================================================

st.set_page_config(page_title="Dispatch & Area Manager Assignment", layout="wide", page_icon="🗺️")

st.title("🗺️ Penugasan Visit Area Manager")

# Menambahkan format tanggal hari ini
tgl_hari_ini = datetime.datetime.now().strftime("%d %B %Y")
st.markdown(f"**Tanggal:** {tgl_hari_ini}")

st.caption("Sistem Rekomendasi Site Visit & Rekapitulasi per Area Manager")

st.sidebar.header("📁 1. Master & Performance Data")
file_master = st.sidebar.file_uploader("1. Master Site Data", type=["csv", "xlsx"])
file_perf = st.sidebar.file_uploader("2. Raw Data BAKTI Daily", type=["csv", "xlsx"])

st.sidebar.markdown("---")
st.sidebar.header("🚫 2. Filter Exclude (Opsional)")
file_parked = st.sidebar.file_uploader("3. Site List Parked", type=["csv", "xlsx"])

st.sidebar.markdown("---")
st.sidebar.header("📅 3. Rentang Evaluasi Daily")
date_start_base = st.sidebar.date_input("Mulai Baseline (Senin)", value=pd.to_datetime("2026-09-14"))
date_end_base = st.sidebar.date_input("Akhir Baseline (Minggu)", value=pd.to_datetime("2026-09-20"))
date_latest = st.sidebar.date_input("Hari Terakhir Data", value=pd.to_datetime("2026-09-24"))

st.sidebar.markdown("---")
help_osrm = "Peta akan lebih cepat jika fitur OSRM ini dimatikan (hanya pakai jarak lurus)."
use_osrm = st.sidebar.checkbox("Gunakan Akses Jalan OSRM", value=False, help=help_osrm)

if file_master is not None and file_perf is not None:
    df_master = pd.read_csv(file_master) if file_master.name.endswith('.csv') else pd.read_excel(file_master)
    df_perf = pd.read_csv(file_perf) if file_perf.name.endswith('.csv') else pd.read_excel(file_perf)

    col_master_site = temukan_kolom(df_master, ['site id', 'site_id', 'siteid', 'site'])
    col_site_name = temukan_kolom(df_master, ['site name', 'sitename', 'nama site', 'site_name'])
    col_kab = temukan_kolom(df_master, ['kabupaten', 'kab/kota', 'kab', 'city'])
    col_kec = temukan_kolom(df_master, ['kecamatan', 'kec', 'district'])
    col_am = temukan_kolom(df_master, ['area manager', 'am', 'area_manager', 'pic am', 'pic'])
    col_lat = temukan_kolom(df_master, ['latitude', 'lat', 'y', 'lat_site'])
    col_lon = temukan_kolom(df_master, ['longitude', 'long', 'lon', 'x', 'long_site'])

    col_perf_site = temukan_kolom(df_perf, ['site id', 'site_id', 'siteid', 'site'])
    col_date = temukan_kolom(df_perf, ['date', 'tanggal', 'datetime', 'waktu', 'day'])
    col_avail = temukan_kolom(df_perf, ['availability (%)', 'availability', 'avail', 'avail (%)'])

    with st.expander("🛠️ Verifikasi Pemetaan Kolom Data", expanded=(col_lat is None or col_lon is None)):
        c1, c2, c3, c4 = st.columns(4)
        col_master_site = c1.selectbox("Site ID (Master):", df_master.columns, index=df_master.columns.get_loc(col_master_site) if col_master_site in df_master.columns else 0)
        col_kab = c2.selectbox("Kabupaten (Master):", df_master.columns, index=df_master.columns.get_loc(col_kab) if col_kab in df_master.columns else 0)
        col_am = c3.selectbox("Area Manager PIC:", df_master.columns, index=df_master.columns.get_loc(col_am) if col_am in df_master.columns else 0)
        col_lat = c4.selectbox("Latitude (Y):", df_master.columns, index=df_master.columns.get_loc(col_lat) if col_lat in df_master.columns else 0)
        
        c5, c6, c7, c8 = st.columns(4)
        col_lon = c5.selectbox("Longitude (X):", df_master.columns, index=df_master.columns.get_loc(col_lon) if col_lon in df_master.columns else 0)
        col_perf_site = c6.selectbox("Site ID (Perf):", df_perf.columns, index=df_perf.columns.get_loc(col_perf_site) if col_perf_site in df_perf.columns else 0)
        col_date = c7.selectbox("Date (Perf):", df_perf.columns, index=df_perf.columns.get_loc(col_date) if col_date in df_perf.columns else 0)
        col_avail = c8.selectbox("Avail % (Perf):", df_perf.columns, index=df_perf.columns.get_loc(col_avail) if col_avail in df_perf.columns else 0)

    df_master['site_id_clean'] = df_master[col_master_site].astype(str).str.strip().str.upper()
    df_perf['site_id_clean'] = df_perf[col_perf_site].astype(str).str.strip().str.upper()
    df_perf['avail_num'] = clean_to_numeric(df_perf[col_avail])

    cw_num = pd.to_datetime(date_start_base).isocalendar().week
    cw_label = f"CW{cw_num}"

    res_aging = process_performance_aging_daily(
        df_perf, 'site_id_clean', col_date, 'avail_num', date_start_base, date_end_base, date_latest
    )
    target_down_sites, list_tambahan_2days, count_w_down, count_achieve_up, count_add_2d = res_aging

    parked_excluded_sites = {}
    if file_parked is not None:
        df_parked = pd.read_csv(file_parked) if file_parked.name.endswith('.csv') else pd.read_excel(file_parked)
        c_parked_site = temukan_kolom(df_parked, ['site id', 'site_id', 'siteid', 'site'])
        c_remark = temukan_kolom(df_parked, ['remark', 'status', 'issue', 'keterangan', 'reason', 'parked'])
        
        if c_parked_site:
            df_parked['site_id_clean'] = df_parked[c_parked_site].astype(str).str.strip().str.upper()
            for _, r in df_parked.iterrows():
                remark_val = str(r[c_remark]).strip() if c_remark and pd.notna(r[c_remark]) else "Parked Site"
                if 'plan visit' not in remark_val.lower():
                    parked_excluded_sites[r['site_id_clean']] = remark_val

    df_assigned = df_master[df_master['site_id_clean'].isin(target_down_sites)].copy()
    df_assigned['kategori_down'] = df_assigned['site_id_clean'].apply(
        lambda x: 'Tambahan (>2 Hari)' if x in list_tambahan_2days else 'Weekly Baseline'
    )

    def check_exclusion(row):
        kab = str(row.get(col_kab, '')).strip().lower()
        site_id = row.get('site_id_clean', '')
        
        if any(rz in kab for rz in REDZONE_KABUPATEN):
            return pd.Series(["EXCLUDE: RedZone Keamanan", False])
        if site_id in parked_excluded_sites:
            return pd.Series([f"EXCLUDE PARKED: {parked_excluded_sites[site_id]}", False])
            
        return pd.Series(["VALID: Eligible Visit", True])

    if not df_assigned.empty:
        res_ex = df_assigned.apply(check_exclusion, axis=1)
        df_assigned['exclusion_reason'] = res_ex[0]
        df_assigned['is_eligible'] = res_ex[1]
    else:
        df_assigned['exclusion_reason'] = []
        df_assigned['is_eligible'] = []

    df_assigned_eligible = df_assigned[df_assigned['is_eligible'] == True].copy()

    # ==============================================================================
    # 4. TAMPILAN OUTPUT 1
    # ==============================================================================
    st.subheader("📊 Output 1: Performance Summary & Assignment Area Manager")
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Input Target Manual")
    target_manual = st.sidebar.number_input(f"Target Perbaikan ({cw_label})", min_value=0, value=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"1. Total Down ({cw_label})", f"{count_w_down} Site", f"Target {target_manual} Site")
    m2.metric("2. Achievement Site Up", f"{count_achieve_up} Site")
    m3.metric("3. Tambahan Down (>2 Hari)", f"{count_add_2d} Site")
    
    total_eligible = len(df_assigned_eligible) if not df_assigned_eligible.empty else 0
    m4.metric("4. Target Visit Eligible", f"{total_eligible} Site")

    st.divider()

    st.markdown("### 🏢 Rekapitulasi Penugasan per Area Manager")
    if not df_assigned.empty:
        df_summary_am = df_assigned.groupby([col_am, col_kab]).agg(
            Total_Target_Down=('site_id_clean', 'count'),
            Site_Down_Tambahan=('kategori_down', lambda x: (x == 'Tambahan (>2 Hari)').sum()),
            Eligible_Visit=('is_eligible', lambda x: x.sum()),
            Excluded_Sites=('is_eligible', lambda x: (~x).sum())
        ).reset_index()

        df_summary_am = df_summary_am.sort_values(by=[col_am, col_kab], ascending=[True, True])
        cols_order = [col_kab, col_am, 'Total_Target_Down', 'Site_Down_Tambahan', 'Eligible_Visit', 'Excluded_Sites']
        df_summary_am = df_summary_am[cols_order]

        st.dataframe(
            df_summary_am.rename(columns={
                col_kab: 'Kabupaten', col_am: 'Area Manager PIC',
                'Total_Target_Down': 'Total Down',
                'Site_Down_Tambahan': 'Tambahan (>2 Hari)',
                'Eligible_Visit': 'Rekomendasi Visit',
                'Excluded_Sites': 'Exclude'
            }),
            use_container_width=True
        )

    st.markdown("### 📋 Detail List Assignment Site Down (Eligible Visit Only)")

    if not df_assigned_eligible.empty:
        df_assigned_eligible = df_assigned_eligible.sort_values(by=[col_am, col_kab, 'site_id_clean'])
        disp_cols = ['site_id_clean']
        if col_site_name: disp_cols.append(col_site_name)
        disp_cols.extend([col_am, col_kab, 'kategori_down'])
        if col_kec: disp_cols.append(col_kec)

        rename_dict = {'site_id_clean': 'Site ID', col_am: 'Area Manager PIC', col_kab: 'Kabupaten', 'kategori_down': 'Kategori Down'}
        if col_site_name: rename_dict[col_site_name] = 'Site Name'
        if col_kec: rename_dict[col_kec] = 'Kecamatan'

        st.dataframe(df_assigned_eligible[disp_cols].rename(columns=rename_dict), use_container_width=True)
    else:
        st.info("Tidak ada site down yang eligible.")

    st.divider()

    # ==============================================================================
    # 5. OUTPUT 2: CLUSTER SITE BERDEKATAN & MAPS (ELIGIBLE ONLY)
    # ==============================================================================
    st.subheader("🛣️ Output 2: Rekomendasi Cluster & Map Sebaran")

    df_assigned_eligible[col_lat] = pd.to_numeric(df_assigned_eligible[col_lat], errors='coerce')
    df_assigned_eligible[col_lon] = pd.to_numeric(df_assigned_eligible[col_lon], errors='coerce')
    df_valid_geo = df_assigned_eligible.dropna(subset=[col_lat, col_lon]).copy()

    if df_valid_geo.empty:
        st.error("⚠️ Data Latitude/Longitude bernilai kosong atau tidak ada site eligible.")
    else:
        st.markdown("### 📦 Export Excel")
        df_all_export = generate_all_am_export_data(
            df_valid_geo, col_am, col_kab, col_kec, col_site_name, col_lat, col_lon, use_osrm
        )

        buffer_all = io.BytesIO()
        with pd.ExcelWriter(buffer_all, engine='openpyxl') as writer:
            df_all_export.to_excel(writer, sheet_name='All_Area_Managers', index=False)
            for am_k, df_group_am in df_all_export.groupby('Area Manager PIC'):
                sheet_title = str(am_k).replace('/', '_').replace('\\', '_')[:31]
                df_group_am.to_excel(writer, sheet_name=sheet_title, index=False)
        buffer_all.seek(0)

        st.download_button(
            label="📥 Download EXCEL Penugasan (Eligible Only)",
            data=buffer_all,
            file_name=f"Report_Penugasan_{cw_label}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("---")

        raw_am_list = sorted(df_valid_geo[col_am].dropna().unique().tolist())
        am_options = ["ALL PIC Area Managers"] + raw_am_list

        selected_am_option = st.selectbox("Pilih PIC Area Manager untuk Map:", am_options)

        if selected_am_option == "ALL PIC Area Managers":
            st.markdown("### 📊 Ringkasan Potensi Cluster")
            summary_am_rows = []
            for am_name in raw_am_list:
                df_am = df_valid_geo[df_valid_geo[col_am] == am_name].copy()
                df_clustered_am = temukan_cluster_berdekatan(df_am, col_lat, col_lon, max_dist_km=30.0)
                df_multi_am = df_clustered_am[df_clustered_am['cluster_id'] != -1]
                
                num_g = df_multi_am['cluster_id'].nunique() if not df_multi_am.empty else 0
                num_s = len(df_multi_am) if not df_multi_am.empty else 0
                c_w = (df_am['kategori_down'] == 'Weekly Baseline').sum()
                c_t = (df_am['kategori_down'] == 'Tambahan (>2 Hari)').sum()
                
                summary_am_rows.append({
                    'Area Manager PIC': am_name,
                    'Total Site Eligible': len(df_am),
                    'Potensi Group': num_g,
                    'Site Dalam Group': num_s,
                    'Weekly Baseline': c_w,
                    'Tambahan (>2 Hari)': c_t
                })

            st.dataframe(pd.DataFrame(summary_am_rows), use_container_width=True)
            st.markdown("### 🗺️ Peta Konsolidasi Sebaran (Clustering Aktif)")
            
            folium_colors = [
                'red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                'lightred', 'darkblue', 'darkgreen', 'cadetblue', 
                'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black'
            ]
            css_colors = {
                'red': '#d33d2a', 'blue': '#38aadd', 'green': '#72b026', 'purple': '#d252b9',
                'orange': '#f69730', 'darkred': '#a23336', 'lightred': '#ff8e7f', 'darkblue': '#0067a3',
                'darkgreen': '#728224', 'cadetblue': '#436978', 'darkpurple': '#5b396b', 'pink': '#ff91ea',
                'lightblue': '#8adaff', 'lightgreen': '#bbf970', 'gray': '#a6a6a6', 'black': '#333333'
            }
            pic_colors = {am: folium_colors[i % len(folium_colors)] for i, am in enumerate(raw_am_list)}
            
            st.markdown("**📌 Visual Key (Warna Marker):**")
            
            legend_cols = st.columns(4)
            col_index = 0
            for am_n, col_n in pic_colors.items():
                hex_col_n = css_colors.get(col_n, 'gray')
                display_text = f"■ {am_n}"
                legend_cols[col_index % 4].markdown(display_text, unsafe_allow_html=True)
                col_index += 1
            
            st.markdown("---")
            
            avg_lat = df_valid_geo[col_lat].mean()
            avg_lon = df_valid_geo[col_lon].mean()
            
            m_all = folium.Map(location=[avg_lat, avg_lon], zoom_start=7, tiles="OpenStreetMap")
            
            for am_name in raw_am_list:
                df_am_map = df_valid_geo[df_valid_geo[col_am] == am_name]
                if df_am_map.empty: continue
                
                mc = plugins.MarkerCluster(name=f"PIC: {am_name}")
                am_color = pic_colors[am_name]
                
                for idx_s, row_s in df_am_map.iterrows():
                    s_id = str(row_s['site_id_clean'])
                    s_nm = str(row_s[col_site_name]) if col_site_name and col_site_name in row_s else ''
                    
                    popup_html = f"[{am_name}] ID: {s_id} - Name: {s_nm}"
                    
                    folium.Marker(
                        location=[row_s[col_lat], row_s[col_lon]], 
                        popup=popup_html,
                        tooltip=s_id,
                        icon=folium.Icon(color=am_color, icon="info-sign")
                    ).add_to(mc)
                
                mc.add_to(m_all)
            
            folium.LayerControl().add_to(m_all)
            st_folium(m_all, width=900, height=550, key="map_all")

        else:
            df_am_s = df_valid_geo[df_valid_geo[col_am] == selected_am_option].copy()

            if not df_am_s.empty:
                df_clustered_am = temukan_cluster_berdekatan(df_am_s, col_lat, col_lon, max_dist_km=30.0)
                df_multi_cluster_am = df_clustered_am[df_clustered_am['cluster_id'] != -1].copy()

                tot_grp = df_multi_cluster_am['cluster_id'].nunique() if not df_multi_cluster_am.empty else 0
                c_sum1, c_sum2, c_sum3, c_sum4 = st.columns(4)
                c_sum1.metric("Total Site Eligible", f"{len(df_am_s)} Site")
                c_sum2.metric("Potensi Cluster", f"{tot_grp} Group")
                c_sum3.metric("Weekly", f"{(df_am_s['kategori_down'] == 'Weekly Baseline').sum()} Site")
                c_sum4.metric("Tambahan", f"{(df_am_s['kategori_down'] == 'Tambahan (>2 Hari)').sum()} Site")

                st.markdown("---")

                if not df_multi_cluster_am.empty:
                    for c_id, group in df_multi_cluster_am.groupby('cluster_id'):
                        st.markdown(f"#### 📍 Group Cluster #{c_id}")
                        first_row = group.iloc[0]
                        curr_lat, curr_lon = first_row[col_lat], first_row[col_lon]

                        g_res = []
                        c_list = []

                        for idx_g, row_g in group.reset_index().iterrows():
                            lat_g, lon_g = row_g[col_lat], row_g[col_lon]
                            c_list.append([lat_g, lon_g])
                            dist_km = hitung_jarak_haversine_vec(curr_lat, curr_lon, pd.Series([lat_g]), pd.Series([lon_g]))[0]
                            
                            has_road = False
                            if use_osrm:
                                has_road, _, _, _ = get_osrm_route(curr_lat, curr_lon, lat_g, lon_g)

                            moda, ket_akses, _ = analisa_moda_per_site(row_g, dist_km, has_road)
                            s_name_val = row_g[col_site_name] if col_site_name and col_site_name in row_g else '-'

                            g_res.append({
                                'Stop #': idx_g + 1,
                                'Site ID': row_g['site_id_clean'],
                                'Site Name': s_name_val,
                                'Kabupaten': row_g[col_kab],
                                'Jarak (KM)': round(dist_km, 2),
                                'Moda Akses': moda
                            })

                        st.dataframe(pd.DataFrame(g_res), use_container_width=True)
                        m_f = folium.Map(location=[group[col_lat].mean(), group[col_lon].mean()], zoom_start=10)

                        for idx_g, row_g in group.reset_index().iterrows():
                            s_id = str(row_g['site_id_clean'])
                            
                            popup_html = f"Stop #{idx_g + 1} - ID: {s_id}"

                            folium.Marker(
                                location=[row_g[col_lat], row_g[col_lon]],
                                popup=folium.Popup(popup_html, max_width=250),
                                tooltip=f"Stop {idx_g + 1}",
                                icon=folium.Icon(color="red", icon="info-sign")
                            ).add_to(m_f)

                        folium.PolyLine(c_list, color="orange", weight=4).add_to(m_f)
                        st_folium(m_f, width=900, height=450, key=f"map_grp_{c_id}")
                        st.markdown("---")
                else:
                    st.info(f"Tidak ada kelompok site berdekatan untuk {selected_am_option}.")
                    m_s = folium.Map(location=[df_am_s[col_lat].mean(), df_am_s[col_lon].mean()], zoom_start=9)
                    
                    mc_s = plugins.MarkerCluster()
                    for idx_s, row_s in df_am_s.reset_index().iterrows():
                        s_id = str(row_s['site_id_clean'])
                        
                        popup_html = f"ID: {s_id}"
                        
                        folium.Marker(
                            location=[row_s[col_lat], row_s[col_lon]],
                            popup=folium.Popup(popup_html, max_width=250),
                            tooltip=s_id,
                            icon=folium.Icon(color="blue", icon="info-sign")
                        ).add_to(mc_s)

                    mc_s.add_to(m_s)
                    st_folium(m_s, width=900, height=450, key=f"map_s_{selected_am_option}")

else:
    st.info("Silakan unggah Master Site Data dan Raw Data BAKTI OSS (Daily) untuk memulai.")