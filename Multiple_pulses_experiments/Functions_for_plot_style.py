def set_plot_properties(ax, style_left_bottom_axis_only = False, xtick_fontsize=5, xlabel_fontsize=7.5, ytick_fontsize=5, ylabel_fontsize=7.5, 
                         title_fontsize=9, axis_linewidth=0.5, plot_linewidth=0.75):
    # Set xtick fontsize
    ax.tick_params(axis='x', labelsize=xtick_fontsize, width = axis_linewidth)

    # Set xlabel fontsize
    ax.set_xlabel(ax.get_xlabel(), fontsize=xlabel_fontsize)

    # Set ytick fontsize
    ax.tick_params(axis='y', labelsize=ytick_fontsize, width = axis_linewidth)

    # Set ylabel fontsize
    ax.set_ylabel(ax.get_ylabel(), fontsize=ylabel_fontsize)

    # Set title fontsize
    ax.set_title(ax.get_title(), fontsize=title_fontsize)

    # Set linewidth for axis lines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(axis_linewidth)

    # Set linewidth for lines inside the plot
    for line in ax.lines:
        line.set_linewidth(plot_linewidth)

    # Leave only bottom and left axis visible    
    if style_left_bottom_axis_only:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)