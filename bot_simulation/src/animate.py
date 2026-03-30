import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

cf=0.0006742675003064854
MAJOR = 81.57/2*cf
MINOR = 44.542/2*cf
Xmax=792*cf
Ymax=718*cf

def construct_ellipsoid(h, k, phi, A=MAJOR, B=MINOR, npoints=20):
    t = np.linspace(0, 2 * np.pi, npoints)
    x = A * np.cos(t)
    y = B * np.sin(t)
    bx = x * np.cos(phi) - y * np.sin(phi) + h
    by = x * np.sin(phi) + y * np.cos(phi) + k
    return bx, by


def animate_sim(data,  filename, resample_freq =1, fps=10):
    df = data[::resample_freq].reset_index()
    #print(df)
    l, r = np.min(df.X) - 0.05, np.max(df.X) + 0.05
    b, t = np.min(df.Y) - 0.05, np.max(df.Y) + 0.05
    ratio = (t - b) / (r - l)
    fig, ax = plt.subplots(figsize=(12, 12 * ratio))

    # Plot elements
    ln1, = plt.plot([], [], c='black', lw=5, animated=True)
    ln2, = plt.plot([], [], c='orange', lw=3,animated=True)
    ln3 = plt.quiver([], [], [], [], scale=1, scale_units='xy', angles='xy', color='blue', animated=True)  # Quiver

    
    text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Initialize the plot
    def init():
        ax.set_xlim(l, r)
        ax.set_ylim(b, t)
        return ln1, ln2, ln3, text
        
    # Update function
    def update(frame):
        h, k, phi = df.iloc[frame][['X', 'Y', 'Theta']]
        tk,vk,wk = df.iloc[frame][['time', 'v1', 'w']]
        X, Y = data.iloc[:frame*resample_freq + 1][['X', 'Y']].values.T
        
        bx, by = construct_ellipsoid(h, k, phi)
        
        ln1.set_data(bx, by)
        ln2.set_data(X, Y)
        ln3.set_offsets(np.array([[h, k]]))
        ln3.set_UVC(np.cos(phi), np.sin(phi))
        
        # Update the text content
        text.set_text(f'Time: {round(tk,4)}\nv1: {round(vk,6)}\nw: {round(wk,6)}')
        
        return ln1, ln2, ln3, text
  
    # Calculate interval in milliseconds
    interval = 1000 / fps
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(df),
                                  init_func=init, blit=True, interval=interval)
    
    # Save the animation as a GIF
    ani.save(filename+'.gif', writer='pillow')
    # Display the animation
    plt.show()

