from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objs as go


# Create animation function with convergence plot
def create_animation(fn, bounds, history, best_values, function_name, algorithm):
    # Generate mesh grid for contour plot
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([fn([xi, yi]) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Optimization Process', 'Convergence Plot'))

    # Add contour plot to the first subplot
    fig.add_trace(
        go.Contour(
            x=x,
            y=y,
            z=Z,
            colorscale='Viridis',
            contours=dict(showlines=False),
            showscale=False,
        ),
        row=1, col=1
    )

    # Prepare frames
    frames = []
    num_frames = len(history)
    for i in range(num_frames):
        positions = np.array(history[i])
        # Scatter plot for particle positions
        scatter = go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode='markers',
            marker=dict(color='red', size=5),
        )
        # Line plot for convergence
        convergence_line = go.Scatter(
            x=list(range(i + 1)),
            y=best_values[:i + 1],
            mode='lines+markers',
            line=dict(color='blue'),
        )
        # Create frame
        frame = go.Frame(
            data=[
                scatter,  # Update particle positions
                convergence_line  # Update convergence plot
            ],
            traces=[1, 2],  # Indices of the traces to update
            name=str(i)
        )
        frames.append(frame)

    # Initial scatter plot for particles
    initial_positions = np.array(history[0])
    fig.add_trace(
        go.Scatter(
            x=initial_positions[:, 0],
            y=initial_positions[:, 1],
            mode='markers',
            name='Particles',
            marker=dict(color='red', size=5),
        ),
        row=1, col=1
    )

    # Initial convergence plot
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[best_values[0]],
            mode='lines+markers',
            name='Best Value',
            line=dict(color='blue'),
        ),
        row=1, col=2
    )

    # Update layout with animation settings
    fig.update_layout(
        title=f'{algorithm.upper()} optimization, benchmark function {function_name.upper()}',
        updatemenus=[dict(
            type='buttons',
            buttons=[dict(label='Play',
                          method='animate',
                          args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])],
            showactive=False,
            y=1.1,
            x=0.5,
            xanchor='center',
            yanchor='top'
        )],
        sliders=[dict(
            steps=[dict(method='animate',
                        args=[[str(k)], dict(mode='immediate', frame=dict(duration=100, redraw=True),
                                             transition=dict(duration=0))],
                        label=str(k)) for k in range(num_frames)],
            transition=dict(duration=0),
            x=0.1,
            xanchor='left',
            y=0,
            yanchor='top'
        )]
    )

    # Assign frames to the figure
    fig.frames = frames

    # Adjust axes ranges and titles
    fig.update_xaxes(title_text='X', range=[bounds[0][0], bounds[0][1]], row=1, col=1)
    fig.update_yaxes(title_text='Y', range=[bounds[1][0], bounds[1][1]], row=1, col=1)
    fig.update_xaxes(title_text='Iteration', range=[0, num_frames], row=1, col=2)
    fig.update_yaxes(title_text='Best Value', range=[min(best_values), max(best_values)], row=1, col=2)

    return fig
