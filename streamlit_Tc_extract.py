import pandas as pd
import ipywidgets as widgets
from ipyfilechooser import FileChooser
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

import streamlit as st


st.set_page_config(layout="wide")
st.title("PPMS Data Plotter & Analyzer")

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

def refine_transition_fit(df, channel, temp_col='Temperature (K)', delta_range=2.0):
    data = df[[temp_col, channel]].dropna()
    T = data[temp_col].values
    R = data[channel].values

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
        delta_T_refine_guess = 0.5
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
    except RuntimeError:
        return None


def process_and_plot(df):
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

    plot_data_vs_time(df)
    plot_data_vs_temperature(df)

def plot_data_vs_time(df):
    global figTime
    figTime = make_subplots(rows=5, cols=1, vertical_spacing=0.0, shared_xaxes=True)

    figTime.add_trace(go.Scatter(x=df['Time Stamp (sec)'], y=df['Temperature (K)']), row=1, col=1)
    figTime.add_trace(go.Scatter(x=df['Time Stamp (sec)'], y=np.round(df['Magnetic Field (Oe)'] / 10000, 1)), row=2, col=1)
    figTime.add_trace(go.Scatter(x=df['Time Stamp (sec)'], y=df['Channel 1 Resistance']), row=3, col=1)
    figTime.add_trace(go.Scatter(x=df['Time Stamp (sec)'], y=df['Channel 2 Resistance']), row=4, col=1)
    figTime.add_trace(go.Scatter(x=df['Time Stamp (sec)'], y=df['Channel 3 Resistance']), row=5, col=1)

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
    figTime.update_xaxes(showticklabels=False, nticks=10, showline=True, linecolor='black', linewidth=1, mirror=True, ticks='inside', spikemode='across+toaxis', spikedash='solid', spikecolor='gray', spikethickness=1)
    figTime.update_yaxes(showline=True, linecolor='black', linewidth=1, mirror=True, ticks='inside')
    figTime.update_xaxes(showticklabels=True, row=5, col=1)
    figTime.update_traces(hovertemplate="%{y}", name="")
    figTime.update_yaxes(title_text="Temperature (K)", row=1, col=1)
    figTime.update_yaxes(title_text="Field (T)", row=2, col=1)
    figTime.update_yaxes(title_text="Ch1 R (Ohm)", row=3, col=1)
    figTime.update_yaxes(title_text="Ch2 R (Ohm)", row=4, col=1)
    figTime.update_yaxes(title_text="Ch3 R (Ohm)", row=5, col=1)

    st.plotly_chart(figTime, use_container_width=True)

def plot_data_vs_temperature(df):
    global figTemp
    figTemp = make_subplots(
        rows=3, cols=1, vertical_spacing=0.03, shared_xaxes=True,
        subplot_titles=("Ch1 Resistance, Ohm", "Ch2 Resistance, Ohm", "Ch3 Resistance, Ohm")
    )

    figTemp.add_trace(go.Scatter(x=df['Temperature (K)'], y=df['Channel 1 Resistance']), row=1, col=1)
    figTemp.add_trace(go.Scatter(x=df['Temperature (K)'], y=df['Channel 2 Resistance']), row=2, col=1)
    figTemp.add_trace(go.Scatter(x=df['Temperature (K)'], y=df['Channel 3 Resistance']), row=3, col=1)

    figTemp.update_layout(
        height=600,
        template="plotly_white",
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1,
                xanchor="center",
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
    figTemp.update_xaxes(showticklabels=True, row=3, col=1)
    figTemp.update_traces(hovertemplate="%{x} K", name="", mode="lines")

    st.plotly_chart(figTemp, use_container_width=True)
        
# ---- FILE UPLOAD AND INITIAL PLOT ----
uploaded_file = st.file_uploader("Choose a data file", type=["csv", "dat", "txt"])

if uploaded_file is not None:
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

    process_and_plot(df)

    # --- ANALYZE BUTTON ---
    if st.button("Analyze"):
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

        for channel in channels:
            for field in fields:
                for sweep_dir in ['cooling', 'warming']:
                    sel = (df['Rounded Field (Oe)'] == field) & (df['Sweep'] == sweep_dir)
                    sweep_df = df[sel]
                    result = refine_transition_fit(sweep_df, channel, temp_col='Temperature (K)', delta_range=delta_range)
                    if result:
                        results[channel][(field, sweep_dir)] = result

        # 6. Plot: All fits and data points for each channel
        for channel in channels:
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
                within_low = np.abs(data_min - R_low) < 0.05 * fit_range
                within_high = np.abs(data_max - R_high) < 0.05 * fit_range
                
                # Plot fit only if R2>=0.9, frac_change>=0.9, and data covers both plateaus
                plot_fit = (R2 >= 0.9) and (frac_change >= 0.9) and within_low and within_high

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
                title=f'<b>{channel}</b>',
                xaxis_title='Temperature (K)',
                yaxis_title='Resistance (Ω)',
                template='plotly_white',
                legend_title='Field/Sweep',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

        # 7. Print summary
        summary_dfs = {}
        for channel in channels:
            rows = []
            for (field, sweep_dir), params in results[channel].items():
                row = {
                    'Field': field,
                    'Sweep': sweep_dir,
                    'Tc': params['Tc'],
                    'Tc_err': params.get('Tc_err', np.nan),
                    'transition_width': params['delta_T'],
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
            within_low = np.abs(summary_df['data_min'] - summary_df['R_low']) < 0.05 * fit_range
            within_high = np.abs(summary_df['data_max'] - summary_df['R_high']) < 0.05 * fit_range
            mask = (summary_df['R2'] >= 0.8) & (summary_df['frac_change'] >= 0.9) & within_low & within_high
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
            with st.expander(f"Summary for {channel}"):
                st.dataframe(summary_df)

            

        # 8.
        #### If you want the 10–90% width, multiply transition_width by 4.394.
        
        for channel in channels:
            df_summary = summary_dfs[channel].copy()
            df_summary = df_summary[df_summary['frac_change'] > 0.9]
            df_summary['Field (T)'] = df_summary['Field'] / 10000
        
            # Only keep cooling for hysteresis to avoid duplicates
            df_hyst = df_summary[df_summary['Sweep'] == 'cooling'].copy()
            df_hyst['abs_hysteresis'] = np.abs(pd.to_numeric(df_hyst['hysteresis'], errors='coerce'))
            df_hyst['abs_hysteresis_err'] = pd.to_numeric(df_hyst['hysteresis_err'], errors='coerce')
        
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Tc vs Field", "Transition Width vs Field", "|Hysteresis| vs Field"),
                shared_xaxes=True
            )
        
            colors = {'cooling': 'blue', 'warming': 'red'}
        
            # 1. Tc vs Field
            for sweep in df_summary['Sweep'].unique():
                mask = df_summary['Sweep'] == sweep
                fig.add_trace(
                    go.Scatter(
                        x=df_summary.loc[mask, 'Field (T)'],
                        y=df_summary.loc[mask, 'Tc'],
                        error_y=dict(type='data', array=df_summary.loc[mask, 'Tc_err']),
                        mode='markers',
                        name=f"Tc {sweep}",
                        marker=dict(color=colors[sweep],size=8),
                        showlegend=True if sweep == 'cooling' else True  # Only show legend once (False)
                    ),
                    row=1, col=1
                )
        
            # 2. Transition width vs Field
            for sweep in df_summary['Sweep'].unique():
                mask = df_summary['Sweep'] == sweep
                fig.add_trace(
                    go.Scatter(
                        x=df_summary.loc[mask, 'Field (T)'],
                        y=df_summary.loc[mask, 'transition_width']*4.394,
                        error_y=dict(type='data', array=df_summary.loc[mask, 'transition_width_err']),
                        mode='markers',
                        name=f"Width {sweep}",
                        marker=dict(color=colors[sweep],size=8),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
            # 3. |hysteresis| vs Field (cooling only)

            # Ensure limits are at least ±0.05
            y_data = df_hyst['abs_hysteresis']
            y_err = df_hyst['abs_hysteresis_err']
            
            # Compute min and max including error bars
            y_min = (y_data - y_err).min()
            y_max = (y_data + y_err).max()
            
            # Ensure limits are at least ±0.05
            y_axis_min = min(-0.05, y_min)
            y_axis_max = max(0.05, y_max)

            fig.add_trace(
                go.Scatter(
                    x=df_hyst['Field (T)'],
                    y=df_hyst['abs_hysteresis'],
                    error_y=dict(type='data', array=df_hyst['abs_hysteresis_err']),
                    mode='markers',
                    name="|Hysteresis|",
                    marker=dict(color='purple',size=8),
                    showlegend=False
                ),
                row=1, col=3
            )
            fig.update_yaxes(range=[y_axis_min, y_axis_max], row=1, col=3)

            font_dict=dict(family='Arial',
               size=12,
               color='black'
               )
        
            # Shared styles
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
            
            # Apply to all subplots (assuming 1 row, 3 columns)
            for col in range(1, 4):
                fig.update_xaxes(row=1, col=col, **xaxis_style)
                fig.update_yaxes(row=1, col=col, **yaxis_style)
            for annotation in fig['layout']['annotations']:
                annotation['y'] += 0.04  # Increase this value as needed (default is ~1.0 for top row)
            
            # Set titles individually
            fig.update_xaxes(title_text="Field (T)", row=1, col=1)
            fig.update_xaxes(title_text="Field (T)", row=1, col=2)
            fig.update_xaxes(title_text="Field (T)", row=1, col=3)
            
            fig.update_yaxes(title_text="Tc (K)", row=1, col=1)
            fig.update_yaxes(title_text="Transition Width (K)", row=1, col=2)
            fig.update_yaxes(title_text="|Hysteresis| (K)", row=1, col=3)
            
            # Layout
            fig.update_layout(
                title_text=f"<b>{channel}</b>",
                template="plotly_white",
                font=font_dict,
                margin=dict(t=80)
            )

            st.plotly_chart(fig, use_container_width=True)


        # Define the model: H(Tc) = Hc2_0 * (1 - Tc/Tc0)
        def H_vs_Tc(Tc, Hc2_0, Tc0):
            return Hc2_0 * (1 - Tc/Tc0)

        n_channels = 3  # Number of channels to plot
        fig, axes = plt.subplots(1, n_channels, figsize=(18, 5), sharey=False, dpi=1800)
        
        for i, channel in enumerate(channels[:n_channels]):
            ax = axes[i]
            df = summary_dfs[channel]  # Use your filtered summary DataFrame for this channel
        
            for sweep_dir, color, marker, ls in zip(
                    ['cooling'],
                    #['cooling', 'warming'],
                    ['blue', 'red'],
                    ['o', 's'],
                    ['-', '--']):
                # Select only this sweep direction
                mask = df['Sweep'] == sweep_dir
                Tc_values = df.loc[mask, 'Tc'].values
                fields_T = df.loc[mask, 'Field'].values / 1e4  # Oe to Tesla
        
                # Remove any NaNs and sort by Tc (for smooth plotting)
                mask_valid = ~np.isnan(Tc_values)
                Tc_values = Tc_values[mask_valid]
                fields_T = fields_T[mask_valid]
                order = np.argsort(Tc_values)
                Tc_values = Tc_values[order]
                fields_T = fields_T[order]
        
                if len(fields_T) > 2:
                    # --- GL (linear) fit ---
                    p0 = [np.max(fields_T), np.max(Tc_values)]
                    popt, pcov = curve_fit(H_vs_Tc, Tc_values, fields_T, p0=p0)
                    Hc2_0_fit, Tc0_fit = popt
                    fields_T_fit_GL = H_vs_Tc(Tc_values, Hc2_0_fit, Tc0_fit)
                    ss_res_GL = np.sum((fields_T - fields_T_fit_GL) ** 2)
                    ss_tot_GL = np.sum((fields_T - np.mean(fields_T)) ** 2)
                    r2_GL = 1 - ss_res_GL / ss_tot_GL
                    Phi_0 = 2.07e-15  # Weber
                    xi_0_GL = np.sqrt(Phi_0 / (2 * np.pi * Hc2_0_fit)) * 1e9  # nm
        
                    ax.plot(Tc_values, fields_T, marker, color=color, label=f'$T_c$ ({sweep_dir})')
                    Tc_fit = np.linspace(0, np.max(Tc_values)*1.05, 200)
                    H_fit = H_vs_Tc(Tc_fit, Hc2_0_fit, Tc0_fit)
                    # First line with main fit info
                    ax.plot(Tc_fit, H_fit, ls, color=color, linewidth=2,
                            label=rf'GL fit: $H_{{c2}}(0)$={Hc2_0_fit:.2f} T')
                    # Second line with extra info, invisible line
                    ax.plot([], [], ' ', label=rf'$\xi(0)$={xi_0_GL:.1f} nm, $R^2$={r2_GL:.3f}')
        
                    # --- Quadratic fit ---
                    pFit = np.polyfit(Tc_values, fields_T, 2)
                    f_quad = np.poly1d(pFit)
                    fields_T_fit_quad = f_quad(Tc_values)
                    ss_res_quad = np.sum((fields_T - fields_T_fit_quad) ** 2)
                    ss_tot_quad = np.sum((fields_T - np.mean(fields_T)) ** 2)
                    r2_quad = 1 - ss_res_quad / ss_tot_quad
                    Tc_quad = np.linspace(0, np.max(Tc_values)*1.05, 200)
                    H_quad = f_quad(Tc_quad)
                    a, b, c = pFit
                    Hc2_0_quad = c
                    xi_0_quad = np.sqrt(Phi_0 / (2 * np.pi * Hc2_0_quad)) * 1e9  # nm
                    ax.plot(Tc_quad, H_quad, 'g--', 
                            label=rf'Quad fit: $H_{{c2}}(0)$={Hc2_0_quad:.2f} T')
                    ax.plot([], [], ' ', 
                            label=rf'$\xi(0)$={xi_0_quad:.1f} nm, $R^2$={r2_quad:.3f}')
        
                    # --- WHH from quadratic fit ---
                    #dHc2dT_at_Tc0 = 2*a*Tc0_fit + b
                    #Hc2_0_quad_WH = -0.69 * Tc0_fit * dHc2dT_at_Tc0
                    #ax.axhline(Hc2_0_quad_WH, color='green', linestyle=':', label=f'WHH (quad): $H_{{c2}}(0)$={Hc2_0_quad_WH:.2f} T')
                else:
                    print(f"{channel} ({sweep_dir}): Not enough points for fit.")
        
            ax.set_title(f'{channel}: $H_{{c2}}$ vs $T_c$')
            ax.set_xlabel('$T_c$ (K)')
            ax.set_ylabel('$H_{c}$ (T)')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)