# Import necessary libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import PSOoptim, DEoptim, SAoptim
import BenchmarkFunctions, Animation

# Initialize the Dash app
app = dash.Dash(__name__)


# Define the app layout
app.layout = html.Div([
    html.H1("Optimization Techniques Visualization"),
    html.Div([
        html.Label("Choose Optimization Algorithm:"),
        dcc.Dropdown(
            id='algorithm',
            options=[
                {'label': 'Differential Evolution (DE)', 'value': 'de'},
                {'label': 'Particle Swarm Optimization (PSO)', 'value': 'pso'},
                # {'label': 'Simulated Annealing (SA)', 'value': 'sa'},
            ],
            value='pso'
        ),
        html.Br(),  # Add a line break
        html.Label("Choose Benchmark Function:"),
        dcc.Dropdown(
            id='function',
            options=[
                {'label': 'Ackley', 'value': 'ackley'},
                {'label': 'Griewank', 'value': 'griewank'},
                {'label': 'Rastrigin', 'value': 'rastrigin'},
                {'label': 'Rosenbrock', 'value': 'rosenbrock'},
                {'label': 'Schwefel 1.2', 'value': 'schwefel_1_2'},
                {'label': 'Sphere', 'value': 'sphere'},
                {'label': 'Weierstrass', 'value': 'weierstrass'},
            ],
            value='sphere'
        ),
        html.Br(),  # Add a line break
        html.Label("Maximum Iterations:"),
        dcc.Input(id='max_iter', type='number', value=50),
        html.Br(), html.Br(),

        html.Button('Run Optimization', id='run-button', n_clicks=0),
    ], style={
        'width': '25%',
        'display': 'inline-block',
        'verticalAlign': 'top'}),
    html.Br(),  # Add a line break

    html.Div([
        dcc.Graph(id='optimization-graph'),
        html.Div(id='result-output'),
    ], style={
        'width': '70%',
        'display': 'inline-block',
        'padding': '0 20'}),
    html.Br(),  # Add a line break

    # Footer Div
    html.Div(
        children=[
            html.P("Created by Michala Jakubcová")  # Replace 'Your Name' with your actual name
        ],
        style={
            'textAlign': 'center',
            'padding': '10px',
            'position': 'fixed',
            'left': '0',
            'bottom': '0',
            'width': '100%',
            'backgroundColor': '#f9f9f9',
            'borderTop': '1px solid #e6e6e6',
            'fontSize': '12px',
            'color': '#777'
        }
    )
])


# Define the callback
@app.callback(
    [Output('optimization-graph', 'figure'),
     Output('result-output', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('algorithm', 'value'),
     State('function', 'value'),
     State('max_iter', 'value')]
)
def run_optimization(n_clicks, algorithm, function_name, max_iter):
    if n_clicks == 0:
        # Initial empty graph and message
        return {}, ''

    # Select the benchmark function
    if function_name == 'rastrigin':
        bounds = [(-5, 5), (-5, 5)]
        fn = BenchmarkFunctions.rastrigin
    elif function_name == 'ackley':
        bounds = [(-5, 5), (-5, 5)]
        fn = BenchmarkFunctions.ackley
    elif function_name == 'sphere':
        bounds = [(-5, 5), (-5, 5)]
        fn = BenchmarkFunctions.sphere
    elif function_name == 'schwefel_1_2':
        bounds = [(-100, 100), (-100, 100)]
        fn = BenchmarkFunctions.schwefel_1_2
    elif function_name == 'rosenbrock':
        bounds = [(-5, 10), (-5, 10)]
        fn = BenchmarkFunctions.rosenbrock
    elif function_name == 'griewank':
        bounds = [(-600, 600), (-600, 600)]
        fn = BenchmarkFunctions.griewank
    elif function_name == 'weierstrass':
        bounds = [(-0.5, 0.5), (-0.5, 0.5)]
        fn = BenchmarkFunctions.weierstrass
    else:
        return {}, 'Invalid function selected.'

    # Run the selected algorithm
    if algorithm == 'pso':
        best_solution, best_value, history, best_values = PSOoptim.pso(fn, bounds, num_particles=30, max_iter=int(max_iter))
    elif algorithm == 'de':
        best_solution, best_value, history, best_values = DEoptim.de(fn, bounds, population_size=50, max_iter=int(max_iter))
    elif algorithm == 'sa':
        best_solution, best_value, history, best_values = SAoptim.sa(fn, bounds, max_iter=int(max_iter))
    else:
        return {}, 'Invalid algorithm selected.'

    # Create the visualization
    fig = Animation.create_animation(fn, bounds, history, best_values, function_name, algorithm)
    # Prepare the result output
    result_text = f'Minimum value found: {best_value:.4f} at position: {best_solution}'
    return fig, result_text


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
