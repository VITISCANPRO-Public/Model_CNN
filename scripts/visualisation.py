#                                          LIBRARIES IMPORT
# ================================================================================================

import plotly.graph_objects as go


#                                     VISUALIZATION
# ================================================================================================

def plot_training_history(history: dict, title: str = "Training History"):
    """Displays the loss and accuracy curves"""
    color_chart = ["#4B9AC7", "#4BE8E0", "#9DD4F3", "#97FBF6"]
    
    # Loss
    fig_loss = go.Figure(data=[
        go.Scatter(
            y=history["loss"],
            name="Training loss",
            mode="lines",
            marker=dict(color=color_chart[0])
        ),
        go.Scatter(
            y=history["val_loss"],
            name="Validation loss",
            mode="lines",
            marker=dict(color=color_chart[1])
        )
    ])
    fig_loss.update_layout(
        title=f'{title} - Loss',
        xaxis_title='Epochs',
        yaxis_title='Cross Entropy Loss'
    )
    fig_loss.write_html(f"{title}_loss.html")
    print(f"Loss plot saved: {title}_loss.html")
    
    # Accuracy
    fig_acc = go.Figure(data=[
        go.Scatter(
            y=history["accuracy"],
            name="Training Accuracy",
            mode="lines",
            marker=dict(color=color_chart[0])
        ),
        go.Scatter(
            y=history["val_accuracy"],
            name="Validation Accuracy",
            mode="lines",
            marker=dict(color=color_chart[1])
        )
    ])
    fig_acc.update_layout(
        title=f'{title} - Accuracy',
        xaxis_title='Epochs',
        yaxis_title='Accuracy'
    )
    fig_acc.write_html(f"{title}_accuracy.html")
    print(f"Accuracy plot saved: {title}_accuracy.html")