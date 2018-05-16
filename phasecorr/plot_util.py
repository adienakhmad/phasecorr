import numpy as np
import pylab as plt


def wiggle_plot(st, figsize, color='black', color_top='red', color_below='blue', normalization_factor=1.0, skipinterval=1, offset = 0):

    st.sort(keys=['starttime'])
    fig, ax = plt.subplots(figsize=figsize)
    maxdata = 0.0
    for i in range(len(st)):
        maxdata = np.max([maxdata, np.max(np.abs(st[i].data))])

    for i in range(len(st)):
        norm_data = st[i].data[::skipinterval] * normalization_factor / maxdata
        times = st[i].times()[::skipinterval] + offset

        # day = dat[i].stats['starttime'].day
        if color == '':
            ax.plot(norm_data + i, times, alpha=0.8)
        else:
            ax.plot(norm_data + i, times, alpha=0.8, color=color)
        # format the ticks
        ax.set_xlabel('Number of Data', fontsize=17)
        ax.set_ylabel('Time', fontsize=17)

        # ax.invert_yaxis()
        ax.set_title('Plot Phase Auto Correlation ' + '\n', fontsize=25)
        ax.fill_betweenx(times, i, norm_data + i, where=(norm_data + i > i), color=color_top)
        ax.fill_betweenx(times, i, norm_data + i, where=(norm_data + i < i), color=color_below)

    print(ax.get_ylim())
    ax.set_ylim(-0.02, ax.get_ylim()[1] + 0.02)
    ax.set_xlim(-2, len(st) + 0.5)
    ax.invert_yaxis()