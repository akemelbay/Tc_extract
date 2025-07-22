import pandas as pd
#import ipywidgets as widgets
#from ipyfilechooser import FileChooser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import math
from scipy import stats
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import plotly.express as px

from scipy.special import expit
import matplotlib.pyplot as plt

import sys
import os
import base64
import random

import streamlit as st

import re

#from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide")

#Hide Streamlit top bar
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stMain .block-container {padding-top: 2rem ! important}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Read and encode the image
with open("qunatum_turkey.png", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()
# Generate random filter values
invert = round(random.uniform(0, 1), 2)  # 0% to 100%
sepia = round(random.uniform(0, 1), 2)
saturate = random.randint(100, 5000)    # 100% to 5000%
hue_rotate = random.randint(0, 360)     # 0deg to 360deg
brightness = round(random.uniform(0.5, 1.5), 2)  # 50% to 150%
contrast = random.randint(50, 150)      # 50% to 150%

filter_str = (
    f"invert({int(invert*100)}%) "
    f"sepia({int(sepia*100)}%) "
    f"saturate({saturate}%) "
    f"hue-rotate({hue_rotate}deg) "
    f"brightness({int(brightness*100)}%) "
    f"contrast({contrast}%)"
)

# Display with random filter
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded}" width="70" 
         style="margin-right: 20px; filter: {filter_str};">
        <span style="font-size: 40px; font-weight: bold;">PPMS Data Plotting & Analysis</span>
    </div>
    """,
    unsafe_allow_html=True
)

#st.title("PPMS Data Plotter & Analyzer")

# ---- PARAMETERS ----
window_length = 11 #savgol window to smoothen T for sweep direction identification
polyorder = 2 #savgol polyorder to smoothen T for sweep direction identification
delta_range = 1.5 #Tc range in [K] to fit transitions with sigmoid

# ---- FUNCTIONS (same as before, omitted here for brevity) ----
def roundMagField(value):
    minField = 0
    maxField = 15e4
    increment = 1e4
    closest_value = min(np.arange(minField, maxField+increment, increment), key=lambda x: abs(x - value))
    return closest_value

def boltzmann_sigmoid(T, R_low, R_high, Tc, delta_T):
    return R_low + (R_high - R_low) * expit((T - Tc) / delta_T)


def refine_transition_fit(df, channel, sweep_dir, field, temp_col='Temperature (K)', delta_range=delta_range, ):
    data = df[[temp_col, channel]].dropna()
    T = data[temp_col].values
    R = data[channel].values

    field = field
    sweep_dir = sweep_dir
    
    if len(T) < 8:
        return None

    R_low_guess = np.min(R)
    R_high_guess = np.max(R)
    Tc_guess = T[np.argmin(np.abs(R - (R_high_guess + R_low_guess)/2))]
    delta_T_guess = delta_range
    p0_initial = [R_low_guess, R_high_guess, Tc_guess, delta_T_guess]
    
    
        
    try:
        popt_initial, _ = curve_fit(boltzmann_sigmoid, T, R, p0=p0_initial)
        Tc_initial = popt_initial[2]

        # Determine available temperature range
        T_min, T_max = T.min(), T.max()
        T_lo = max(T_min, Tc_initial - delta_range)
        T_hi = min(T_max, Tc_initial + delta_range)

        mask_refine = (T >= T_lo) & (T <= T_hi)
        T_refine = T[mask_refine]
        R_refine = R[mask_refine]

        if len(T_refine) < 5:
            return None
        
        
        R_low_refine_guess = np.min(R_refine)
        R_high_refine_guess = np.max(R_refine)
        delta_T_refine_guess = delta_range
        p0_refine = [R_low_refine_guess, R_high_refine_guess, Tc_initial, delta_T_refine_guess]

        popt_refine, pcov_refine = curve_fit(boltzmann_sigmoid, T_refine, R_refine, p0=p0_refine)
        R_low_fit, R_high_fit, Tc_fit, delta_T_fit = popt_refine

        # --- Parameter uncertainties ---
        perr = np.sqrt(np.diag(pcov_refine))
        R_low_err, R_high_err, Tc_err, delta_T_err = perr

        # --- R2 calculation ---
        R_pred = boltzmann_sigmoid(T_refine, R_low_fit, R_high_fit, Tc_fit, delta_T_fit)
        ss_res = np.sum((R_refine - R_pred) ** 2)
        ss_tot = np.sum((R_refine - np.mean(R_refine)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        resistance_change = np.abs(R_high_fit - R_low_fit)
        frac_change = resistance_change / np.abs(R_high_fit) if np.abs(R_high_fit) > 0 else 0

        return {
            'R_low': R_low_fit,
            'R_high': R_high_fit,
            'Tc': Tc_fit,
            'delta_T': delta_T_fit,
            'T_refine': T_refine,
            'R_refine': R_refine,
            'R2': R2,
            'frac_change': frac_change,
            'Tc_err': Tc_err,
            'delta_T_err': delta_T_err
        }
        
    except RuntimeError as e:
        #st.write(f"curve_fit failed for {channel}, {field} Oe, {sweep_dir}: {e}")
        # Return error message instead of writing to Streamlit
        return {'error': f"curve_fit failed for {channel}, {field} Oe, {sweep_dir}: {e}"}

 # Shared styles
font_dict=dict(family='Arial',
       size=12,
       color='black'
       )
xaxis_style = dict(
    showline=True,
    showgrid=True,
    showticklabels=True,
    linecolor='black',
    linewidth=2,
    ticks='outside',
    tickfont=font_dict,
    mirror=True,
    tickwidth=2,
    tickcolor='black',
    title_font=font_dict
)

yaxis_style = dict(
    showline=True,
    showgrid=True,
    showticklabels=True,
    linecolor='black',
    linewidth=2,
    ticks='outside',
    tickfont=font_dict,
    mirror=True,
    tickwidth=2,
    tickcolor='black',
    title_font=font_dict
)

def process_and_plot(df,channel_list_from_fileName):
    # Ensure required columns are present
    required_columns = [
        'Time Stamp (sec)', 'Temperature (K)', 'Magnetic Field (Oe)',
        'Channel 1 Resistance', 'Channel 2 Resistance', 'Channel 3 Resistance', 'Field Status (code)'
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing in the data file.")

    df['Time Stamp (sec)'] = pd.to_datetime(df['Time Stamp (sec)'], unit='s', utc=True)
    df['Elapsed Time'] = df['Time Stamp (sec)'].diff().fillna(pd.Timedelta(seconds=0))
    df['Channel 1 Resistance'] = df['Channel 1 Resistance'].replace('-----', 'NaN').astype('float64')

    channel_list_from_fileName = channel_list_from_fileName
    st.header("Raw data as a function of time", divider=False)
    plot_data_vs_time(df,channel_list_from_fileName)
    st.header("Raw data as a function of temperature", divider=False)
    plot_data_vs_temperature(df,channel_list_from_fileName)

def plot_data_vs_time(df,channel_list_from_fileName):
    channel_list_from_fileName = channel_list_from_fileName
    global figTime
    figTime = make_subplots(rows=5, cols=1, vertical_spacing=0.05, shared_xaxes=True, subplot_titles=[
        #"Temperature (K)",
        #"Magnetic Field (T)",
        #channel_list_from_fileName[0],
        #channel_list_from_fileName[1],
        #channel_list_from_fileName[2]
    ])

    figTime.add_trace(go.Scatter(x=df['Time Stamp (sec)'], y=df['Temperature (K)'], name="T, (K)"), row=1, col=1)
    figTime.add_trace(go.Scatter(x=df['Time Stamp (sec)'], y=np.round(df['Magnetic Field (Oe)'] / 10000, 1), name="H, (T)"), row=2, col=1)
    figTime.add_trace(go.Scatter(x=df['Time Stamp (sec)'], y=df['Channel 1 Resistance'], name = channel_list_from_fileName[0]), row=3, col=1)
    figTime.add_trace(go.Scatter(x=df['Time Stamp (sec)'], y=df['Channel 2 Resistance'], name = channel_list_from_fileName[1]), row=4, col=1)
    figTime.add_trace(go.Scatter(x=df['Time Stamp (sec)'], y=df['Channel 3 Resistance'], name = channel_list_from_fileName[2]), row=5, col=1)

    figTime.update_layout(
        height=1000,
        template="plotly_white",
        hovermode="x",
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=False,
                x=1,
                xanchor="right",
                y=1,
                yanchor="top",
                buttons=[
                    dict(label="Lin/Log", method="relayout", args=[{"yaxis.type": "linear"}], args2=[{"yaxis.type": "log"}]),
                ]
            )
        ]
    )

    
    figTime.update_xaxes(showticklabels=False, nticks=10, showline=True, linecolor='black', linewidth=1, mirror=True, ticks='inside', spikemode='across+toaxis', spikedash='solid', spikecolor='gray', spikethickness=1,tickformat="%H:%M:%S")
    figTime.update_yaxes(showline=True, linecolor='black', linewidth=1, mirror=True, ticks='inside')
    figTime.update_xaxes(showticklabels=True, title_text="Time", row=5, col=1)
    #figTime.update_traces(hovertemplate="%{y}", name="")
    figTime.update_yaxes(title_text="Temperature (K)", row=1, col=1)
    figTime.update_yaxes(title_text="Field (T)", row=2, col=1)
    figTime.update_yaxes(title_text="Ch1 R (Ω)", row=3, col=1)
    figTime.update_yaxes(title_text="Ch2 R (Ω)", row=4, col=1)
    figTime.update_yaxes(title_text="Ch3 R (Ω)", row=5, col=1)

    # Apply to all subplots (assuming 1 col, 5 rows)
    for row in range(1, 6):
        figTime.update_xaxes(row=row, col=1, **xaxis_style)
        figTime.update_yaxes(row=row, col=1, **yaxis_style)
        
    # Hide x-axis tick labels for all but the last subplot
    for row in range(1, 5):  # For rows 1-4 (not the last row)
        figTime.update_xaxes(showticklabels=False, row=row, col=1)

    #Add right margin to fix right border
    figTime.update_layout(margin=dict(r=20))
        
    st.plotly_chart(figTime, config = {'displayModeBar': True}, use_container_width=True)

def plot_data_vs_temperature(df,channel_list_from_fileName):
    channel_list_from_fileName = channel_list_from_fileName
    global figTemp
    figTemp = make_subplots(
        rows=3, cols=1, vertical_spacing=0.1, shared_xaxes=True,
        subplot_titles=(
            #f"Ch1 Resistance, Ohm ({channel_list_from_fileName[0]})", f"Ch2 Resistance, Ohm ({channel_list_from_fileName[1]})", f"Ch3 Resistance, Ohm ({channel_list_from_fileName[2]})"
        )
    )

    figTemp.add_trace(go.Scatter(x=df['Temperature (K)'], y=df['Channel 1 Resistance'], name = channel_list_from_fileName[0]), row=1, col=1)
    figTemp.add_trace(go.Scatter(x=df['Temperature (K)'], y=df['Channel 2 Resistance'], name = channel_list_from_fileName[1]), row=2, col=1)
    figTemp.add_trace(go.Scatter(x=df['Temperature (K)'], y=df['Channel 3 Resistance'], name = channel_list_from_fileName[2]), row=3, col=1)

    figTemp.update_layout(
        height=800,
        template="plotly_white",
        showlegend=False,
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                #pad={"r": 50, "t": 10},
                showactive=True,
                x=1,
                xanchor="right",
                y=1,
                yanchor="top",
                buttons=[
                    dict(label="300 K", method="relayout", args=[{"xaxis.range": (df['Temperature (K)'].min(), 300)}]),
                    dict(label="20 K", method="relayout", args=[{"xaxis.range": (df['Temperature (K)'].min(), 20)}]),
                    dict(label="15 K", method="relayout", args=[{"xaxis.range": (df['Temperature (K)'].min(), 15)}]),
                ]
            )
        ]
    )
    figTemp.update_xaxes(showticklabels=False, spikemode='across+toaxis', spikedash='solid', spikecolor='gray', spikethickness=1)
    figTemp.update_xaxes(showticklabels=True, title_text="Temperature (K)", row=3, col=1)
    #figTemp.update_traces(hovertemplate="%{x} K", name="", mode="lines")

    figTemp.update_yaxes(title_text="Ch1 R (Ω)", row=1, col=1)
    figTemp.update_yaxes(title_text="Ch2 R (Ω)", row=2, col=1)
    figTemp.update_yaxes(title_text="Ch3 R (Ω)", row=3, col=1)
    
    for row in range(1, 4):
        figTemp.update_xaxes(row=row, col=1, **xaxis_style)
        figTemp.update_yaxes(row=row, col=1, **yaxis_style)

    #Add right margin to match figTime that uses margin to fix right border
    figTemp.update_layout(margin=dict(r=20))

    st.plotly_chart(figTemp, config = {'displayModeBar': True}, use_container_width=True)

# ---- FILE UPLOAD AND INITIAL PLOT ----
uploaded_file = st.file_uploader("", type=["dat"])

#Extract sample IDs from filename
def extract_channel_info(filename):
    filename = re.sub(r'\.(dat|csv|txt)$', '', filename, flags=re.IGNORECASE)
    filename_wo_date = filename.split('_', 1)[-1]
    blocks = re.split(r'(?=_(?:CH\d|ch\d)[-_])', '_' + filename_wo_date)
    result = {}
    for block in blocks:
        match = re.match(r'_(CH\d|ch\d)[-_](.+)', block)
        if match:
            ch, info = match.groups()
            result[ch.lower()] = info
    return result

def channel_info_to_list(channel_dict, n_channels=3):
    ch_list = [""] * n_channels
    for ch, info in channel_dict.items():
        num = int(re.search(r'\d+', ch).group())
        if 1 <= num <= n_channels:
            ch_list[num-1] = info
    return ch_list

def sci_not(val, precision=2):
    """Returns a string in scientific notation for LaTeX."""
    if val == 0:
        return f"{0:.{precision}f}"
    exp = int(np.floor(np.log10(abs(val))))
    coeff = val / 10**exp
    return f"{coeff:.{precision}f}\\times 10^{{{exp}}}"

if uploaded_file is not None:

    filename = uploaded_file.name
    channel_info = extract_channel_info(filename)
    channel_list_from_fileName = channel_info_to_list(channel_info)
    #st.write("Channel info dictionary:")
    #st.write(channel_info)
    st.subheader("Sample IDs")
    #st.write(channel_list_from_fileName)

    cols = st.columns(3)
    ch1 = cols[0].text_input("Channel 1:", value=channel_list_from_fileName[0])
    ch2 = cols[1].text_input("Channel 2:", value=channel_list_from_fileName[1])
    ch3 = cols[2].text_input("Channel 3:", value=channel_list_from_fileName[2])

    channel_list_from_fileName = [ch1, ch2, ch3]

    
    # Parse file only once!
    if 'df' not in st.session_state or st.session_state['file_name'] != uploaded_file.name:
        lines = uploaded_file.getvalue().decode(errors='replace').splitlines()
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() == '[Data]':
                data_start = i + 1
                break
        import io
        df = pd.read_csv(io.StringIO('\n'.join(lines[data_start:])))
        st.session_state['df'] = df
        st.session_state['file_name'] = uploaded_file.name
    else:
        df = st.session_state['df']

    process_and_plot(df,channel_list_from_fileName)

    
    with st.form("analysis_form"):
        st.header("Analyze data")
            
        delta_range = st.number_input(
            "Transition window to fit ± (K)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            format="%.2f"
        )
        frac_change_input = st.number_input(
            "Min. fractional resistance change",
            min_value=0.0,
            max_value=1.5,
            value=0.9,
            step=0.1,
            format="%.2f"
        )
        R2_requirement_input = st.number_input(
            "R2 requirement",
            min_value=0.0,
            max_value=1.0,
            value=0.99,
            step=0.001,
            format="%.3f"
        )
    
        submitted = st.form_submit_button("Analyze")
        
        if submitted:
            
            # 1. Filter for T < 20 K
            df = st.session_state['df']
            df = df[df['Temperature (K)'] < 20].copy()
    
            # 2. Smooth temperature
            df['Temperature (K) Smoothed'] = savgol_filter(df['Temperature (K)'], window_length=window_length, polyorder=polyorder, mode='interp')
    
            # 3. Add rounded field column
            df['Rounded Field (Oe)'] = df['Magnetic Field (Oe)'].apply(roundMagField)
    
            # 4. Assign sweep direction per field using smoothed temperature
            df['Sweep'] = pd.Series(dtype='object')
            for field in df['Rounded Field (Oe)'].unique():
                mask = df['Rounded Field (Oe)'] == field
                field_df = df.loc[mask].sort_values('Time Stamp (sec)')
                dT = np.diff(field_df['Temperature (K) Smoothed'], prepend=np.nan)
                sweep = np.where(dT < 0, 'cooling', 'warming')
                df.loc[field_df.index, 'Sweep'] = sweep
    
            # 5. Analyze and store results
            channels = ['Channel 1 Resistance', 'Channel 2 Resistance', 'Channel 3 Resistance']
            fields = sorted(df['Rounded Field (Oe)'].unique())
            results = {channel: {} for channel in channels}

            #Prepare for any error messages to receive from refine_transition_fit()
            error_messages = []
            for channel in channels:
                for field in fields:
                    for sweep_dir in ['cooling', 'warming']:
                        sel = (df['Rounded Field (Oe)'] == field) & (df['Sweep'] == sweep_dir)
                        sweep_df = df[sel]
                        result = refine_transition_fit(sweep_df, channel, sweep_dir, field, temp_col='Temperature (K)', delta_range=delta_range)
                        if result:
                            if 'error' in result:
                                error_messages.append(result['error'])
                            else:
                                results[channel][(field, sweep_dir)] = result
            
            channels = [ch for ch in channels if len(results[ch]) > 0]
            if error_messages:
                with st.expander(":orange[Fitting errors]",icon=":material/error:"):
                    for msg in error_messages:
                        st.write(msg)


           
            # 6. Plot: All fits and data points for each channel
            for idx, channel in enumerate(channels):
                fig = go.Figure()
                for (field, sweep_dir), params in results[channel].items():
                    Tc = params['Tc']
                    delta_T = params['delta_T']
                    R_low = params['R_low']
                    R_high = params['R_high']
                    T_refine = params['T_refine']
                    R_refine = params['R_refine']
                    R2 = params['R2']
                    frac_change = params['frac_change']
                    
                    # Assign color by sweep direction
                    if sweep_dir == 'cooling':
                        color = 'blue'
                        line_dash = 'solid'
                    else:
                        color = 'red'
                        line_dash = 'solid'
    
                    # Data coverage check
                    data_min = R_refine.min()
                    data_max = R_refine.max()
                    fit_range = np.abs(R_high - R_low)
                    #Check whether fit plateaued, ensuring that measured data actually covers both ends of the transition
                    within_low = np.abs(data_min - R_low) < 0.05 * fit_range
                    within_high = np.abs(data_max - R_high) < 0.1 * fit_range
                    
                    # Plot fit only if R2>=0.9, frac_change>=0.9, and data covers both plateaus
                    plot_fit = (R2 >= R2_requirement_input) and (frac_change >= frac_change_input) and within_low and within_high
    
                    # Data points (used for fit)
                    fig.add_trace(go.Scatter(
                        x=T_refine,
                        y=R_refine,
                        mode='markers',
                        marker=dict(color=color, size=5, symbol='circle'),
                        name=f'{field/10000:.2f} T {sweep_dir} data'
                    ))
    
                    if plot_fit:
                        # Fitted curve
                        T_fit = np.linspace(T_refine.min(), T_refine.max(), 200)
                        R_fit = boltzmann_sigmoid(T_fit, R_low, R_high, Tc, delta_T)
                        fig.add_trace(go.Scatter(
                            x=T_fit,
                            y=R_fit,
                            mode='lines',
                            line=dict(color=color, width=3, dash=line_dash),
                            name=f'{field/10000:.2f} T {sweep_dir} fit (Tc={Tc:.3f}K, R²={R2:.3f})'
                        ))

                    
    
                fig.update_layout(
                    title=f'<b>{channel}</b> ({channel_list_from_fileName[idx]})',
                    xaxis_title='Temperature (K)',
                    yaxis_title='Resistance (Ω)',
                    template='plotly_white',
                    legend_title='Field/Sweep',
                    height=600
                )
                fig.update_xaxes(xaxis_style)
                fig.update_yaxes(yaxis_style)
                st.plotly_chart(fig, use_container_width=True)

            

                
    
            # 7. Print summary
            summary_dfs = {}
            for idx, channel in enumerate(channels):
                rows = []
                for (field, sweep_dir), params in results[channel].items():
                    row = {
                        'Field': field,
                        'Sweep': sweep_dir,
                        'Tc': params['Tc'],
                        'Tc_err': params.get('Tc_err', np.nan),
                        'transition_width': params['delta_T']*4.394,
                        'transition_width_err': params.get('delta_T_err', np.nan),
                        'R2': params['R2'],
                        'frac_change': params['frac_change'],
                        'hysteresis': None,
                        'hysteresis_err': None,
                        'R_low': params['R_low'],
                        'R_high': params['R_high'],
                        'data_min': params['R_refine'].min(),
                        'data_max': params['R_refine'].max(),
                    }
                    rows.append(row)
                    
                summary_df = pd.DataFrame(rows)
                
                # Data coverage filter
                fit_range = np.abs(summary_df['R_high'] - summary_df['R_low'])
                #Check whether fit plateaued, ensuring that measured data actually covers both ends of the transition
                within_low = np.abs(summary_df['data_min'] - summary_df['R_low']) < 0.05 * fit_range
                within_high = np.abs(summary_df['data_max'] - summary_df['R_high']) < 0.1 * fit_range
                mask = (summary_df['R2'] >= R2_requirement_input) & (summary_df['frac_change'] >= frac_change_input) & within_low & within_high
                summary_df = summary_df[mask].copy()
                
    
                # Calculate hysteresis and its error
                for field in summary_df['Field'].unique():
                    mask_cooling = (summary_df['Field'] == field) & (summary_df['Sweep'] == 'cooling')
                    mask_warming = (summary_df['Field'] == field) & (summary_df['Sweep'] == 'warming')
                    if mask_cooling.any() and mask_warming.any():
                        Tc_cooling = summary_df.loc[mask_cooling, 'Tc'].values[0]
                        Tc_warming = summary_df.loc[mask_warming, 'Tc'].values[0]
                        Tc_cooling_err = summary_df.loc[mask_cooling, 'Tc_err'].values[0]
                        Tc_warming_err = summary_df.loc[mask_warming, 'Tc_err'].values[0]
                        hysteresis = Tc_cooling - Tc_warming
                        hysteresis_err = np.sqrt(Tc_cooling_err**2 + Tc_warming_err**2)
                        summary_df.loc[mask_cooling, 'hysteresis'] = hysteresis
                        summary_df.loc[mask_warming, 'hysteresis'] = hysteresis
                        summary_df.loc[mask_cooling, 'hysteresis_err'] = hysteresis_err
                        summary_df.loc[mask_warming, 'hysteresis_err'] = hysteresis_err
                summary_dfs[channel] = summary_df
                with st.expander(f"Summary for {channel} ({channel_list_from_fileName[idx]})",icon=":material/table_view:"):
                    st.dataframe(summary_df)
                
            st.session_state['summary_dfs'] = summary_dfs
            st.session_state['channels'] = channels

            #Print XY data
            with st.expander("XY Data per sweep",icon=":material/backup_table:"):  # Main expander per channel
                 for idx, channel in enumerate(channels):
                    with st.expander(f"Data for {channel} ({channel_list_from_fileName[idx]})",icon=":material/chart_data:"):  # Main expander per channel
                        for (field, sweep_dir), params in results[channel].items():
                            T_refine = params['T_refine']
                            R_refine = params['R_refine']
                
                            with st.expander(f"Field: {field} Oe, sweep: {sweep_dir}",icon=":material/line_start:"):  # Sub-expander for each sweep
                                df_xydata = pd.DataFrame({
                                    'Temperature, K': T_refine,
                                    'Resistance, Ohm': R_refine
                                })
                                st.dataframe(df_xydata,hide_index=True,selection_mode="multi-column")
    
                
    
    # 8.
    
    if 'summary_dfs' in st.session_state:
        summary_dfs = st.session_state['summary_dfs']
        channels = st.session_state['channels']
    
        
        # Define the model: H(Tc) = Hc2_0 * (1 - Tc/Tc0)
        def H_vs_Tc(Tc, Hc2_0, Tc0):
            return Hc2_0 * (1 - Tc/Tc0)
        
        Phi_0 = 2.07e-15  # Weber
        
        for idx, channel in enumerate(channels):
            # Prepare summary df as before
            df_summary = summary_dfs[channel].copy()
            df_summary['Field (T)'] = df_summary['Field'] / 10000
            df_summary = df_summary[(df_summary['frac_change'] > frac_change_input) & (df_summary['Sweep'] == 'cooling')]
            
        
            df_hyst = df_summary[(df_summary['Sweep'] == 'cooling') & (df_summary['frac_change'] > frac_change_input)].copy()
            df_hyst['abs_hysteresis'] = np.abs(pd.to_numeric(df_hyst['hysteresis'], errors='coerce'))
            df_hyst['abs_hysteresis_err'] = pd.to_numeric(df_hyst['hysteresis_err'], errors='coerce')
        
            figAnalysis = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Tc vs Field", "Transition Width vs Field", "Hysteresis vs Field"),
                shared_xaxes=True
            )
        
            colors = {'cooling': 'blue', 'warming': 'red'}
            
            # 1. Tc vs Field (and fits)
            for sweep in df_summary['Sweep'].unique():
                mask = df_summary['Sweep'] == sweep
                x = df_summary.loc[mask, 'Tc'].round(3).values
                y = df_summary.loc[mask, 'Field (T)'].values
                xerr = df_summary.loc[mask, 'Tc_err'].round(3).values
                yerr = None # If you have field errors, put here
                # Scatter points
                figAnalysis.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        error_x=dict(type='data', array=xerr),
                        mode='markers',
                        name=f'Tc ({sweep})',
                        marker=dict(color=colors[sweep], size=8),
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
                # --- FITTING LOGIC (GL and quadratic) ---
                # Only fit if enough points
                mask_valid = ~np.isnan(x)
                x_fit = x[mask_valid]
                y_fit = y[mask_valid]
                if len(x_fit) > 2:
                    # Sort by x for plotting fits
                    order = np.argsort(x_fit)
                    x_fit = x_fit[order]
                    y_fit = y_fit[order]
        
                    # GL fit
                    p0 = [np.max(y_fit), np.max(x_fit)]
                    try:
                        popt, pcov = curve_fit(H_vs_Tc, x_fit, y_fit, p0=p0)
                        Hc2_0_fit, Tc0_fit = popt
                        y_fit_GL = H_vs_Tc(x_fit, Hc2_0_fit, Tc0_fit)
                        ss_res_GL = np.sum((y_fit - y_fit_GL) ** 2)
                        ss_tot_GL = np.sum((y_fit - np.mean(y_fit)) ** 2)
                        r2_GL = 1 - ss_res_GL / ss_tot_GL
                        xi_0_GL = np.sqrt(Phi_0 / (2 * np.pi * Hc2_0_fit)) * 1e9  # nm
        
                        # Plot GL fit line
                        Tc_fit_line = np.linspace(0, np.max(x_fit)*1.05, 200)
                        H_fit_line = H_vs_Tc(Tc_fit_line, Hc2_0_fit, Tc0_fit)
                        figAnalysis.add_trace(
                            go.Scatter(
                                x=Tc_fit_line, y=H_fit_line,
                                mode='lines',
                                line=dict(color=colors[sweep], dash='solid', width=2),
                                name=f'GL fit',
                                showlegend=True
                            ),
                            row=1, col=1
                        )
                    except Exception as e:
                        st.write(f"GL fit failed for {channel} ({sweep}): {e}")
        
                    # Quadratic fit
                    try:
                        pFit = np.polyfit(x_fit, y_fit, 2)
                        f_quad = np.poly1d(pFit)
                        y_fit_quad = f_quad(x_fit)
                        ss_res_quad = np.sum((y_fit - y_fit_quad) ** 2)
                        ss_tot_quad = np.sum((y_fit - np.mean(y_fit)) ** 2)
                        r2_quad = 1 - ss_res_quad / ss_tot_quad
                        a, b, c = pFit
                        Hc2_0_quad = c
                        xi_0_quad = np.sqrt(Phi_0 / (2 * np.pi * Hc2_0_quad)) * 1e9  # nm
                        Tc_quad = np.linspace(0, np.max(x_fit)*1.05, 200)
                        H_quad = f_quad(Tc_quad)
                        figAnalysis.add_trace(
                            go.Scatter(
                                x=Tc_quad, y=H_quad,
                                mode='lines',
                                line=dict(color='green', dash='dash', width=2),
                                name=f'Quad fit',
                                showlegend=True
                            ),
                            row=1, col=1
                        )
                    except Exception as e:
                        st.write(f"Quadratic fit failed for {channel} ({sweep}): {e}")
        
            # 2. Transition width vs Field
            for sweep in df_summary['Sweep'].unique():
                mask = df_summary['Sweep'] == sweep
                figAnalysis.add_trace(
                    go.Scatter(
                        x=df_summary.loc[mask, 'Field (T)'],
                        y=df_summary.loc[mask, 'transition_width'],
                        error_y=dict(type='data', array=df_summary.loc[mask, 'transition_width_err']),
                        mode='markers',
                        name=f"Width {sweep}",
                        marker=dict(color=colors[sweep],size=8),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
            # 3. |hysteresis| vs Field (cooling only)
            y_data = df_hyst['abs_hysteresis']
            y_err = df_hyst['abs_hysteresis_err']
            y_min = (y_data - y_err).min()
            y_max = (y_data + y_err).max()
            y_axis_min = min(-0.05, y_min)
            y_axis_max = max(0.05, y_max)
        
            figAnalysis.add_trace(
                go.Scatter(
                    x=df_hyst['Field (T)'],
                    y=df_hyst['hysteresis'],
                    error_y=dict(type='data', array=df_hyst['hysteresis_err']),
                    mode='markers',
                    name="Hysteresis",
                    marker=dict(color='purple',size=8),
                    showlegend=False
                ),
                row=1, col=3
            )
            figAnalysis.update_yaxes(range=[y_axis_min, y_axis_max], row=1, col=3)
        
            # Axis and layout formatting
            for col in range(1, 4):
                figAnalysis.update_xaxes(row=1, col=col, **xaxis_style)
                figAnalysis.update_yaxes(row=1, col=col, **yaxis_style)
            for annotation in figAnalysis['layout']['annotations']:
                annotation['y'] += 0.04
        
            figAnalysis.update_xaxes(title_text="Tc (K)", row=1, col=1)
            figAnalysis.update_xaxes(title_text="Tc (K)", row=1, col=2)
            figAnalysis.update_xaxes(title_text="Tc (K)", row=1, col=3)
            figAnalysis.update_yaxes(title_text="Field (T)", row=1, col=1)
            figAnalysis.update_yaxes(title_text="Transition width (K)", row=1, col=2)
            figAnalysis.update_yaxes(title_text="Hysteresis (K)", row=1, col=3)
        
            figAnalysis.update_layout(
                title_text=f"<b>{channel}</b> ({channel_list_from_fileName[idx]})",
                template="plotly_white",
                font=font_dict,
                margin=dict(t=80)
            )
            
            st.plotly_chart(figAnalysis, use_container_width=True)
            if not df_summary.empty:
                # --- FIT SUMMARY: Hc2(0) and xi(0) ---
                #st.write("**Ginzburg-Landau (GL) fit:**")
                st.latex(
                    f"\\textbf{{Ginzburg-Landau (GL) fit: }} H_{{c2}}(0) = {Hc2_0_fit:.2f}\\,\\text{{T}},\\quad "
                    f"\\xi(0) = {xi_0_GL:.1f}\\,\\text{{nm}},\\quad "
                    f"R^2 = {r2_GL:.3f}"
                )
                #st.markdown("**Quadratic fit:**")
                st.latex(
                    f"\\textbf{{Quadratic fit: }} H_{{c2}}(0) = {Hc2_0_quad:.2f}\\,\\text{{T}},\\quad "
                    f"\\xi(0) = {xi_0_quad:.1f}\\,\\text{{nm}},\\quad "
                    f"R^2 = {r2_quad:.3f}"
                )
                # --- SUPERCONDUCTING GAP ---
                if not df_summary[df_summary['Field'] == 0].empty:
                    hbar = 1.055e-34  # J·s
                    kB_eV = 8.617e-5  # eV/K
                    mu_0 = 1.25663706e-6  # H/m
            
                    Tc_0_field = df_summary.loc[df_summary['Field'] == 0, 'Tc'].values[0]
                    gap_0T = 1.76 * Tc_0_field * kB_eV  # eV
                    gap_0T_J = gap_0T * 1.602e-19  # J
            
                    #st.markdown("**Superconducting gap at 0 T:**")
                    st.latex(
                        f"\\textbf{{Superconducting gap at 0 T: }} \\Delta(0) = {gap_0T*1e3:.2f}\\,\\text{{meV}}"
                    )
                    # --- KINETIC INDUCTANCE ---
                    R_high = df_summary.loc[df_summary['Field'] == 0, 'R_high'].values[0]
                    R_sq = 4.532 * R_high
                    Lk = hbar * R_sq / (np.pi * gap_0T_J)  # H/sq
                    Lk_pH_sq = Lk * 1e12  # pH/sq
            
                    #st.markdown("**Kinetic inductance per square:**")
                    st.latex(
                        f"\\textbf{{Kinetic inductance: }} L_k = {Lk_pH_sq:.2f}\\,\\text{{pH}}/\\square"
                        f" \\quad (R_{{\\text{{high}}}}^{{\\text{{fit}}}} = {R_high:.2f}\\,\\Omega,\\;"
                        f"R_{{\\square}} = {R_sq:.2f}\\,\\Omega/\\square)"
                    )
            
                    # --- PENETRATION DEPTH ---
                    t_nm = st.number_input("Enter film thickness in nm:", min_value=0, value=200, step = 10, key=f"thickness_{channel}")
                    t_m = t_nm * 1e-9
                    if st.button("Calculate penetration depth", key=f"penetration_btn_{channel}"):
                        lambda_m = np.sqrt(Lk * t_m / mu_0)
                        lambda_nm = lambda_m * 1e9
                        #st.markdown("**London penetration depth:**")
                        st.latex(
                            f"\\textbf{{London penetration depth: }} \\lambda = {lambda_nm:.1f}\\,\\text{{nm}}"
                        )
            
                    # --- DERIVATION/FORMULAS COLLAPSIBLE ---
                    with st.expander("Show derivations and formulas"):
                        st.latex(r"H_{c2}(0) = \text{GL fit parameter (extrapolated upper critical field at } T=0\text{)}")
                        st.latex(r"\xi(0) = \sqrt{\frac{\Phi_0}{2\pi H_{c2}(0)}}")
                        st.latex(r"\Delta(0) = 1.76\,k_B T_c")
                        st.latex(r"L_k = \frac{\hbar R_{\square}}{\pi \Delta(0)}")
                        st.latex(r"\lambda = \sqrt{\frac{L_{\square}\, t}{\mu_0}}")
                        # If you want, you can also add a line with the numeric substitutions here
                        st.latex(
                            f"\\Delta(0) = 1.76 \\times {Tc_0_field:.2f}\\,\\text{{K}} "
                            f"\\times {kB_eV:.5f}\\,\\text{{eV/K}} = {gap_0T*1e3:.2f}\\,\\text{{meV}}"
                        )
                        st.latex(
                            f"L_k = \\frac{{{sci_not(hbar)}\\,\\text{{J}}\\cdot\\text{{s}} \\times {R_sq:.2f}\\,\\Omega}}"
                            f"{{\\pi \\times {sci_not(gap_0T_J)}\\,\\text{{J}}}} = {Lk_pH_sq:.2f}\\,\\text{{pH}}/\\square"
                        )
                        if 'lambda_nm' in locals():
                            st.latex(
                                f"\\lambda = \\sqrt{{\\frac{{{sci_not(Lk)}\\,\\text{{H}}/\\square \\times {t_nm:.1f}\\,\\text{{nm}}}}"
                                f"{{{sci_not(mu_0)}\\,\\text{{H/m}}}}}} = {lambda_nm:.1f}\\,\\text{{nm}}"
                            )
                else:
                    Tc_0_field = None  # or np.nan, or raise an error, etc.

                #st.dataframe(df_summary)
# Inject custom CSS to left-align KaTeX
st.markdown("""
    <style>
    .katex-display {
        text-align: left !important;
    }
    </style>
""", unsafe_allow_html=True)
# Inject custom CSS to set input width
st.markdown("""
        <style>
        div[data-testid="stNumberInputContainer"] {
            width: 150px;
        }
        </style>
    """, unsafe_allow_html=True)
# Inject custom CSS to set Error icon color to orange
st.markdown("""
    <style>
    [data-testid="stExpanderIconError"] {
        color: rgb(217, 90, 0) !important;
    }
    </style>
""", unsafe_allow_html=True)