import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
import matplotlib
import json
import base64

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Ladle Tearout",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Ladle Tearout Report")
st.markdown("Enter values in the grid below. The 3D diagram will update automatically.")
# disable built in deploy button
st.set_option("client.toolbarMode", "viewer")

rows = [str(i) for i in range(1, 33)]
# Columns 12, 1, 2... 11 (Clock positions)
cols = ["12"] + [str(i) for i in range(1, 12)]

# ONE-TIME URL STATE LOAD
if "initial_load_complete" not in st.session_state:
    st.session_state.initial_load_complete = True

    # Defaults
    st.session_state.r_val = 20.0
    st.session_state.e_val = 30
    st.session_state.a_val = 300
    st.session_state.df_data = pd.DataFrame(np.nan, index=rows, columns=cols)

    if "data" in st.query_params:
        try:
            decoded = base64.b64decode(st.query_params["data"])
            url_data = json.loads(decoded.decode())

            st.session_state.r_val = float(url_data.get("r", 20.0))
            st.session_state.e_val = int(url_data.get("e", 30))
            st.session_state.a_val = int(url_data.get("a", 300))

            if "vals" in url_data:
                st.session_state.df_data = pd.DataFrame(
                    url_data["vals"], index=rows, columns=cols
                )

            # Consume URL once
            st.query_params.clear()

        except Exception:
            pass

# SIDEBAR CONTROLS
st.sidebar.header("Configuration")

# Radius
ladle_radius = st.sidebar.slider("Ladle Radius (R)", min_value=5.0, max_value=50.0, key="r_val", step=1.0)

# Camera View
st.sidebar.subheader("Camera View")
elevation_angle = st.sidebar.slider("Viewing Elevation", min_value=-90, max_value=90, key="e_val")
azimuth_angle = st.sidebar.slider("Rotation", min_value=0, max_value=360, key="a_val")

# Plot Title
plot_title = st.sidebar.text_input("Plot Title", value="Ladle Tearout Report")

st.sidebar.subheader("Color Scale")
vmin = st.sidebar.number_input("Min Value (Red)", value=1.0)
vmax = st.sidebar.number_input("Max Value (Green)", value=6.0)

# DATA INPUT 

# Reset session state if columns changed (e.g. during development/updating code)
if 'df_data' in st.session_state:
    if list(st.session_state.df_data.columns) != cols:
        del st.session_state['df_data']

# Initialize session state for data if not present
if 'df_data' not in st.session_state:
    # Create a DataFrame with NaNs (empty)
    st.session_state.df_data = pd.DataFrame(np.nan, index=rows, columns=cols)

# Display Data Editor
st.markdown("### Brick Measurements")
edited_df = st.data_editor(
    st.session_state.df_data,
    height=300,
    width='stretch',
    num_rows="fixed"
)

# Convert dataframe to numpy array for processing
# Replace empty strings or non-numeric with NaN
data_numeric = edited_df.apply(pd.to_numeric, errors='coerce').values

# --- Generating Share Link ---
st.sidebar.markdown("---")
st.sidebar.subheader("Sharing")

if st.sidebar.button("Generate Share Link"):
    # Create the payload
    save_data = {
        "r": ladle_radius,
        "e": elevation_angle,
        "a": azimuth_angle,
        "vals": edited_df.to_dict()
    }
    json_str = json.dumps(save_data)
    b64_str = base64.b64encode(json_str.encode()).decode()
    
    # Construction of a proper link. 
    # In Streamlit Cloud environments, st.query_params alone won't give the domain.
    # This approach works across both local and cloud deployments.
    # Note: We use a clickable markdown link as requested.
    try:
        # Construct the URL by combining the current base path with the data param
        share_url = f"?data={b64_str}"
        
        st.sidebar.success("Link Generated!")
        st.sidebar.markdown("Right-click the link below and select **'Copy Link Address'** to share:")
        st.sidebar.markdown(f"### [Share Design Link]({share_url})")
        st.sidebar.info("When the recipient opens this link, your measurements and camera view will load automatically.")
    except Exception as e:
        st.sidebar.error("Could not generate link.")

# MATH & HELPERS 

# ----- CONSTANTS -----
n_rows, n_cols = data_numeric.shape # Should be 32, 12
dz = 1.0          # brick height
w_t_const = 1.0   # tangential width
dr_min = 1.0      # min radial thickness
dr_max = 6.0      # max radial thickness
gap_above = 0.8   # space for labels

# Normalize and Color Map
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = matplotlib.colormaps.get_cmap("RdYlGn")
dr_scale = (dr_max - dr_min) / max(1e-9, (vmax - vmin))

# Labels
if n_cols == 12:
    hour_labels_master = ["12"] + [str(h) for h in range(1, 12)]
else:
    hour_labels_master = [f"C{c+1}" for c in range(n_cols)]

def unit_vectors(theta):
    """Return (tangent, radial, vertical) unit vectors at angle theta."""
    r_hat = np.array([np.cos(theta), np.sin(theta), 0.0])        # outward (radial)
    t_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])       # CCW tangent
    z_hat = np.array([0.0, 0.0, 1.0])                            # up
    return t_hat, r_hat, z_hat

def box_faces_outer_aligned(R, z_center, w_t, dr, t_hat, r_hat, z_hat):
    """
    Build faces for a brick whose OUTWARD (+R) face lies exactly on the circle at radius R.
    """
    base_out_center = R * r_hat + (z_center - dz/2.0) * z_hat
    top_out_center  = R * r_hat + (z_center + dz/2.0) * z_hat
    dt = w_t / 2.0

    out_base = [base_out_center + (+dt)*t_hat, base_out_center + (-dt)*t_hat]
    out_top  = [top_out_center  + (+dt)*t_hat, top_out_center  + (-dt)*t_hat]

    in_base = [p - dr * r_hat for p in out_base]
    in_top  = [p - dr * r_hat for p in out_top]

    top_face    = [out_top[0], out_top[1], in_top[1],  in_top[0]]
    bottom_face = [out_base[0], out_base[1], in_base[1], in_base[0]]
    plusR_face  = [out_base[0], out_base[1], out_top[1], out_top[0]]   # outward/front
    minusT_face = [out_base[1], in_base[1],  in_top[1],  out_top[1]]
    minusR_face = [in_base[1],  in_base[0],  in_top[0],  in_top[1]]    # inward/back
    plusT_face  = [in_base[0],  out_base[0], out_top[0], in_top[0]]
    faces = [top_face, bottom_face, plusR_face, minusT_face, minusR_face, plusT_face]
    return faces, plusR_face

def render_fixed_positions(ax, data_arr, active_cols, active_labels, R):
    """
    Render columns at their fixed clock positions (no auto spacing).
    """
    # 12 o'clock is usually at +Y axis (pi/2) in math, moving clockwise
    angles_all = np.pi/2 - np.arange(n_cols) * (2 * np.pi / max(n_cols, 1))

    all_faces      = []
    all_facecolors = []

    for c in active_cols:
        theta = angles_all[c]
        t_hat, r_hat, z_hat = unit_vectors(theta)

        for r in range(n_rows-1, -1, -1):
            val_raw = data_arr[r, c]
            
            # Validation
            if pd.isna(val_raw):
                continue
            try:
                val = float(val_raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue

            # Calculate Thickness based on value
            dr  = float(np.clip(dr_min + dr_scale * (val - vmin), dr_min, dr_max))
            w_t = w_t_const
            z_center = (n_rows - 1 - r) * dz + dz/2.0

            faces, _ = box_faces_outer_aligned(R, z_center, w_t, dr, t_hat, r_hat, z_hat)
            color = cmap(norm(val))

            all_faces.extend(faces)
            all_facecolors.extend([color] * len(faces))

    if not all_faces:
        return False

    poly = Poly3DCollection(
        all_faces,
        facecolors=all_facecolors,
        edgecolors='0.5',
        linewidths=0.3,
        zsort='average' # Simple z-sorting
    )
    ax.add_collection3d(poly)

    # Hour labels
    for c, lbl in zip(active_cols, active_labels):
        th = angles_all[c]
        pos = R * np.array([np.cos(th), np.sin(th), 0.0])
        ax.text(pos[0], pos[1], n_rows * dz + gap_above, lbl,
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Camera, limits, style
    ax.view_init(elev=elevation_angle, azim=azimuth_angle)
    pad = dr_max + 1.0
    ax.set_xlim(-R - pad, R + pad)
    ax.set_ylim(-R - pad, R + pad)
    ax.set_zlim(0, n_rows * dz + 1.6)

    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.set_axis_off()
    ax.grid(False)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    
    return True

# MAIN EXECUTION

# Active Column Detection
active_cols = [c for c in range(n_cols) if np.isfinite(data_numeric[:, c]).any()]
active_count = len(active_cols)
active_labels = [hour_labels_master[c] for c in active_cols]

col1, col2 = st.columns([1, 4])

with col1:
    st.info("💡 **Instructions**\n\n1. Use the table above to enter values.\n2. Columns 12-11 represent positions 12 through 11 on a clock face.\n3. Empty cells are skipped.\n4. Adjust Radius and View in the sidebar.")

    if st.button("Load Sample Data"):
        # Fill columns 12, 3, 6, 9 with random values for all rows
        demo_data = pd.DataFrame(np.nan, index=rows, columns=cols)
        
        # Indices in 'cols': 12->0, 3->3, 6->6, 9->9
        target_indices = [0, 3, 6, 9]
        
        for col_idx in target_indices:
            # Generate random values between 2.0 and 4.5 for all 32 rows
            # This provides variety in color and thickness
            values = np.random.uniform(1, 5, size=len(rows))
            demo_data.iloc[:, col_idx] = values
            
        st.session_state.df_data = demo_data
        st.rerun()

    if st.button("Clear Data"):
        st.session_state.df_data = pd.DataFrame(np.nan, index=rows, columns=cols)
        st.rerun()

with col2:
    if active_count == 0:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No brick values found.\nAdd numeric values to the grid above.",
                ha="center", va="center", fontsize=14, color="gray")
        st.pyplot(fig)
    else:
        # Create Plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        success = render_fixed_positions(ax, data_numeric, active_cols, active_labels, ladle_radius)
        
        ax.set_title(plot_title, fontsize=15, pad=20)
        
        # Colorbar
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.04)
        cbar.set_label("Thickness Value", fontsize=10)
        
        st.pyplot(fig)

