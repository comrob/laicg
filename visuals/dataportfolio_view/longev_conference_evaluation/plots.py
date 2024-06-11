from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np

class ReferenceAndMap():
    def __init__(
            self, sensory_labels: List[str], magnitude_label: str, sensory_ids: List[int], y_lims: List[Tuple[float, float]],
            label_size: float, tick_size: float, x_cut: Tuple[float, float], time_label: str, map_cmap: str, map_box_limits: List[float],
            map_labels = List[str]
            ) -> None:
        self.sensory_labels = sensory_labels
        self.magnitude_label = magnitude_label
        self.y_lims = y_lims
        self.label_size = label_size
        self.tick_size = tick_size
        self.x_cut = x_cut
        self.time_label = time_label
        self.map_cmap = map_cmap
        self.map_box_limits = map_box_limits
        self.map_labels = map_labels

    def reference_and_map_plot(self,
            fig, t_axis_seg, t_axis_ite, y_trg_signal, y_obs_signal, legs_magnitude_signal, distance_signal, heading_signal, location_signal, goal_point):
        # plt.rcParams["figure.figsize"] = (COLUMN_WIDTH, COLUMN_WIDTH/3)
        # f = fig()
        gs = GridSpec(3, 2, figure=fig, width_ratios=[0.6, 0.35], wspace=0.03, hspace=0.25)
        axs = [[fig.add_subplot(gspec)] for gspec in [gs[0, 0], gs[1, 0], gs[2, 0]]]
        """EVOLUTION"""
        """sensors"""
        # axs = V.common.subplots(f, len(sel_sensors) + 1, 1)
        for i in range(len(y_obs_signal)):
            ax = axs[i][0]
            ax.plot(t_axis_seg, y_trg_signal[i], label="Reference", color="r", alpha=0.9)
            ax.plot(t_axis_seg, y_obs_signal[i], label="Estimation", color="b", alpha=0.9)
        """magnitudes"""
        ax_magnitude = axs[-1][0]
        ## servo magnitude
        ax_magnitude.plot(t_axis_seg, legs_magnitude_signal, linestyle="--", color="k", alpha=0.9)
        ax_magnitude.set_ylim(self.y_lims[-1])
        ## distance from goal
        ax_magnitude.fill_between(t_axis_ite, 0, 0.5, color="r", alpha=0.1)
        ax_magnitude.plot(t_axis_ite, distance_signal, color="k", alpha=0.9)

        # styling
        side_labels = [self.sensory_labels[i] for i in range(len(y_trg_signal))] + [self.magnitude_label]
        for i in range(len(axs)):
            ax = axs[i][0]
            # add vertical line in the middle of the cut
            ax.set_ylabel(side_labels[i], fontsize=self.label_size)
            # ax.grid()
            ax.set_xlim(np.min(t_axis_seg), np.max(t_axis_seg))
            ax.axvline(x=(t_axis_seg[0]+t_axis_seg[-1])/2, color="r", linestyle="--", alpha=0.5)

            ax.set_ylim(self.y_lims[i])
            ## set y ticks
            # arange array of y ticks between the limits with 2.5 step
            ticks = np.arange(self.y_lims[i][0], self.y_lims[i][1]+.01, 2.5)
            ax.set_yticks([i for i in ticks])
            # every odd ticklabel is empty string
            ax.set_yticklabels([str(int(ticks[i])) if i%2 == 0 else "" for i in range(len(ticks))])

            ax.tick_params(axis='both', which='major', labelsize=self.tick_size)
            ax.get_yaxis().set_label_coords(-0.1,0.5)

            if i < len(axs) - 1:
                ax.set_xticklabels([])

        axs[-1][0].set_xlabel(self.time_label, fontsize=self.label_size)
        
        """MAP"""
        ax = fig.add_subplot(gs[:, 1])

        # scatter location with the gradient coloring
        cmap = plt.cm.get_cmap(self.map_cmap)
        ax.scatter(location_signal[:, 0], location_signal[:, 1], c=[cmap(i) for i in heading_signal], alpha=1, s=4)
        # show goal with a circle of radius 0.5
        ax.add_patch(plt.Circle((goal_point[0], goal_point[1]), 0.5, color="r", alpha=0.5))
        # add large white goal marker
        ax.plot([goal_point[0]], [goal_point[1]], linestyle='', marker='o', markersize=10, label="Goal", color="w", alpha=0.9)

        ## styling
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_xlim(self.map_box_limits[0], self.map_box_limits[1])
        ax.set_ylim(self.map_box_limits[2], self.map_box_limits[3])
        ax.set_xticks([i for i in range(0, self.map_box_limits[1]+1)])
        ax.set_xticklabels([str(i) for i in range(0, self.map_box_limits[1]+1)])

        ax.set_yticks([i for i in range(self.map_box_limits[2], self.map_box_limits[3]+1)])
        ax.tick_params(axis='both', which='major', labelsize=self.tick_size)

        # ax.set_xticklabels([str(i) for i in range(-5, 6)])
        ax.set_xlabel(self.map_labels[0], fontsize=self.label_size)
        ax.set_ylabel(self.map_labels[1], fontsize=self.label_size)
        ax.grid()
        #grey background
        ax.set_facecolor('grey')


class SensoriMotorHists():
    def __init__(self, motor_label: str, sensor_label: str,
                 error_lim, motor_lim,
                 sensori_labels,
                 motor_labels,weights_norm,
                 tick_size, label_size, weight_cmap
                 ) -> None:
        self.motor_label = motor_label
        self.sensor_label = sensor_label
        self.error_lim=error_lim
        self.motor_lim=motor_lim
        self.sensori_labels=sensori_labels
        self.motor_labels=motor_labels
        self.weights_norm=weights_norm
        self.tick_size=tick_size
        self.label_size=label_size
        self.weight_cmap=weight_cmap


    def sensori_motor_hists(self, fig, motor_amplitude_value, performance_error_value, model_weights):
        gs = GridSpec(2, 2, figure=fig, width_ratios=[0.3, 0.7], wspace=0.03, hspace=0.05)
        ax_model = fig.add_subplot(gs[1,1])
        ax_motor_hist = fig.add_subplot(gs[0,1])
        ax_error_hist = fig.add_subplot(gs[1,0])

        ##
        ax_model.matshow(model_weights, cmap=self.weight_cmap, aspect="auto", norm=self.weights_norm)
        ax_motor_hist.stairs(motor_amplitude_value, color="k", fill=True, alpha=0.9)
        ax_error_hist.stairs(performance_error_value, color="r", orientation="horizontal", fill=True)
        
        
        ##
        ax_motor_hist.set_xlim(0, len(motor_amplitude_value))
        ax_motor_hist.set_ylabel(self.motor_label, fontsize=self.label_size)
        ax_motor_hist.set_xticks([i + 0.5 for i in range(len(motor_amplitude_value))])
        ax_motor_hist.set_xticklabels([])
        ax_motor_hist.yaxis.tick_right()
        ax_motor_hist.yaxis.set_label_position("right")
        ax_motor_hist.set_ylim(self.motor_lim)
        ax_motor_hist.tick_params(labelsize=self.tick_size) 



        ax_error_hist.set_ylim(0, len(performance_error_value))
        ax_error_hist.set_xlim(self.error_lim)
        ## rotate ax_error_hist x axis
        ax_error_hist.invert_xaxis()
        ax_error_hist.invert_yaxis()
        ax_error_hist.set_yticks([i + 0.5 for i in range(len(performance_error_value))])
        ax_error_hist.set_yticklabels([])
        ax_error_hist.yaxis.tick_right()
        ax_error_hist.set_xlabel(self.sensor_label, fontsize=self.label_size)
        ax_error_hist.tick_params(labelsize=self.tick_size) 

        ## ax_model xticks
        ax_model.set_xticks([i for i in range(len(model_weights[0]))])
        ax_model.tick_params(axis='x', rotation=70, labelsize=self.tick_size) 
        ax_model.tick_params(axis='y', labelsize=self.tick_size) 
        ax_model.set_xticklabels([self.motor_labels[i] for i in range(len(model_weights[0]))])
        ax_model.set_yticks([i for i in range(len(performance_error_value))])
        ax_model.set_yticklabels([self.sensori_labels[i] for i in range(len(performance_error_value))])
        ax_model.yaxis.tick_right()
        ax_model.xaxis.tick_bottom()





