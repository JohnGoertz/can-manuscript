import gumbi as gmb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle
from scipy.integrate import trapz
from scipy.optimize import minimize
from scipy.stats import linregress
from sklearn.metrics import confusion_matrix


class Objective:
    def __init__(self, gp, ds, patients, x_offset, LR_file):
        self.gp = gp
        self.ds = ds
        self.patients = patients
        self.limits = None
        self.x_offset = x_offset
        self.EMRI_offset = 0
        self.IFI44L_offset = 0
        self.EMRI_tilt = 0
        self.IFI44L_tilt = 0
        self.scale = 1
        self.vshift = 0
        self.sig = None
        self.EMRI_grid = None
        self.IFI44L_grid = None
        self.nEMRI = None
        self.nIFI44L = None
        self.LR_Scores = pd.read_csv(LR_file)
        self.patient_scores = None
        self.all_scores = None
        self.reg = None
        self.roc = {}

        self.extent = np.max(np.abs(ds.wide["FAM-HEX"]))
        self.norm = mpl.colors.Normalize(vmin=-self.extent, vmax=+self.extent)
        self.cmap = sns.diverging_palette(20, 220, as_cmap=True)
        self.palette = sns.diverging_palette(20, 220, n=2)

        self.set_limits()

    def EMRI_objective(self, x):
        return (2.208 * (x + self.x_offset) - 10.843) * self.scale

    def IFI44L_objective(self, x):
        return (-1.723 * (x + self.x_offset) + 9.258) * self.scale

    @property
    def objective(self):
        assert self.EMRI_grid is not None
        assert self.IFI44L_grid is not None

        return (
            self.EMRI_objective(self.EMRI_grid).values()
            + self.IFI44L_objective(self.IFI44L_grid).values()
        )

    def set_limits(self, pad=0.1):
        mn, mx = [
            self.ds.wide["IFI44L Copies"].min(),
            self.ds.wide["IFI44L Copies"].max(),
        ]
        IFI44L = 2 * pad * (mx - mn) * np.array([-1, 1]) + np.array([mn, mx])

        mn, mx = [self.ds.wide["EMRI Copies"].min(), self.ds.wide["EMRI Copies"].max()]
        EMRI = 2 * pad * (mx - mn) * np.array([-1, 1]) + np.array([mn, mx])

        self.limits = self.gp.parray(**{"IFI44L Copies": IFI44L, "EMRI Copies": EMRI})

    def predict(self):
        limits = self.gp.parray(
            **{
                "IFI44L Copies": self.limits["IFI44L Copies"] + self.IFI44L_offset,
                "EMRI Copies": self.limits["EMRI Copies"] + self.EMRI_offset,
            }
        )
        XY = self.gp.prepare_grid(limits=limits)

        self.sig = self.gp.predict_grid(with_noise=False)
        XY = self.gp.prepare_grid(limits=self.limits)

        self.EMRI_grid = XY["EMRI Copies"]
        self.IFI44L_grid = XY["IFI44L Copies"]

        self.nEMRI = (
            (self.EMRI_grid - np.min(self.EMRI_grid))
            / (np.max(self.EMRI_grid) - np.min(self.EMRI_grid))
            * 2
            - 1
        ).values()
        self.nIFI44L = (
            (self.IFI44L_grid - np.min(self.IFI44L_grid))
            / (np.max(self.IFI44L_grid) - np.min(self.IFI44L_grid))
            * 2
            - 1
        ).values()

        self.sig = (
            self.sig
            - self.nEMRI * self.EMRI_tilt
            - self.nIFI44L * self.IFI44L_tilt
            + self.vshift
        )

        return self.sig

    def plot_predictions(self, ax=None, colorbar=True):
        ax = plt.gca() if ax is None else ax

        plt.sca(ax)
        pp = gmb.ParrayPlotter(
            x=self.IFI44L_grid, y=self.EMRI_grid, z=self.sig / self.scale
        )
        pp(plt.contourf, cmap=self.cmap, norm=self.norm, zorder=-10)
        if colorbar:
            pp.colorbar(ax=ax)
        pp(plt.contour, levels=0, colors="k", norm=self.norm, zorder=-5)

    def plot_deviation(self, ax=None, colorbar=True):
        ax = plt.gca() if ax is None else ax

        plt.sca(ax)
        pp = gmb.ParrayPlotter(
            x=self.IFI44L_grid, y=self.EMRI_grid, z=self.sig.μ - self.objective
        )
        pp(plt.contourf, cmap=self.cmap, norm=self.norm, zorder=-10)
        if colorbar:
            pp.colorbar(ax=ax)

    def plot_observations(self, ax=None, **kwargs):
        ax = plt.gca() if ax is None else ax

        g = sns.scatterplot(
            data=self.ds.wide,
            ax=ax,
            x="IFI44L Copies",
            y="EMRI Copies",
            hue="FAM-HEX",
            hue_norm=self.norm,
            palette=self.cmap,
            **({"s": 20 ** 2, "zorder": 0, "legend": False} | kwargs),
        )

    def plot_patients(self, ax=None, **kwargs):
        ax = plt.gca() if ax is None else ax

        sns.scatterplot(
            data=self.patients,
            x="IFI44L",
            y="EMRI",
            hue="Diagnosis",
            hue_order=["Viral", "Bacterial"],
            palette=self.palette,
            ax=ax,
            **({"zorder": 10, "legend": False} | kwargs),
        )

    def plot_all(self, figsize=(12, 5), patients=True, observations=True):
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        self.plot_predictions(ax=axs[0])
        self.plot_deviation(ax=axs[1])

        if observations:
            self.plot_observations(ax=axs[0])

        if patients:
            self.plot_patients(ax=axs[0])
            self.plot_patients(ax=axs[1])

        plt.tight_layout()

    def plot_marginals(self, figsize=(12, 4)):
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        plt.sca(axs[0])
        x = self.gp.grid_vectors["EMRI Copies"].flatten()
        gmb.ParrayPlotter(x, self.sig.mean(axis=1)).plot()
        gmb.ParrayPlotter(x, self.EMRI_objective(x).values()).plot(
            line_kws={"ls": "--"}
        )

        plt.sca(axs[1])
        x = self.gp.grid_vectors["IFI44L Copies"].flatten()
        gmb.ParrayPlotter(x, self.sig.mean(axis=0)).plot()
        gmb.ParrayPlotter(x, self.IFI44L_objective(x).values()).plot(
            line_kws={"ls": "--"}
        )

        plt.tight_layout()

    @property
    def RMSE(self):
        assert self.sig is not None
        return np.sqrt(np.mean(np.square(self.sig.μ - self.objective)))

    @property
    def nRMSE(self):
        assert self.sig is not None
        return self.RMSE / (self.extent * 2)

    def optimize_scale_shift(self, show=True):
        #         assert self.sig is not None

        def fun(scale_shift):
            scale, shift = scale_shift
            self.scale = scale
            self.vshift = shift
            self.predict()

            return self.RMSE

        rslt = minimize(fun, (self.scale, self.vshift))
        if show:
            print("Scale/shift:", *[f"\t{x:0.4f}" for x in rslt.x])
            print("RMSE:", f"{rslt.fun:0.4f}")
            print("nRMSE:", f"{rslt.fun/(self.extent*2):0.1%}")
            print()

        return rslt

    def optimize_marginal(self, gene, param, show=True, x0=0):
        assert self.sig is not None

        x = self.gp.grid_vectors[f"{gene} Copies"].flatten()
        axis = {"IFI44L": 0, "EMRI": 1}[gene]

        def fun(arg):

            setattr(self, f"{gene}_{param}", arg[0])
            self.predict()
            objective = getattr(self, f"{gene}_objective")(x).values()

            return np.sqrt(np.mean(np.square(self.sig.mean(axis=axis).μ - objective)))

        rslt = minimize(fun, x0)
        if show:
            print(f"{gene} {param}: {rslt.x[0]:0.4f}")
            print("RMSE:", f"{rslt.fun:0.4f}")
            print("nRMSE:", f"{rslt.fun/(self.extent*2):0.1%}")
            print()

        return rslt

    def score_patients(self):
        patient_parray = self.gp.parray(
            **{
                "EMRI Copies": self.patients["EMRI"].values,
                "IFI44L Copies": self.patients["IFI44L"].values,
            }
        )

        patient_predictions = (
            self.gp.predict_points(patient_parray, with_noise=True) / self.scale
        )

        self.patient_scores = self.patients[
            ["Sample", "EMRI", "IFI44L", "Diagnosis"]
        ].assign(
            GP_Score_l=(patient_predictions.μ - patient_predictions.σ2),
            GP_Score_m=(patient_predictions.μ),
            GP_Score_u=(patient_predictions.μ + patient_predictions.σ2),
        )

        self.all_scores = self.LR_Scores.drop(
            columns=["EMRI", "IFI44L", "FAM89A"]
        ).merge(
            self.patient_scores.drop(columns=["EMRI", "IFI44L"]),
            on=["Sample", "Diagnosis"],
        )

        self.reg = linregress(self.all_scores.LR_Score_m, self.all_scores.GP_Score_m)

    def build_roc(self, lvl="m"):
        self.roc[lvl] = []
        scores = self.patient_scores[f"GP_Score_{lvl}"]
        dxs = self.patient_scores["Diagnosis"]
        thresholds = np.hstack(
            [scores.min() - 1, sorted(scores.values), scores.max() + 1]
        )
        for thresh in thresholds:
            tn, fp, fn, tp = confusion_matrix(dxs == "Viral", scores <= thresh).ravel()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            self.roc[lvl].append([tpr, fpr])

        self.roc[lvl] = np.array(self.roc[lvl])
        return self.roc[lvl]

    def plot_roc(self, lvl="m", ax=None, **kwargs):
        ax = plt.gca() if ax is None else ax
        tpr_values, fpr_values = zip(*self.roc[lvl])
        ax.plot(fpr_values, tpr_values, **kwargs)
        ax.set_ylabel("Sensitivity")
        ax.set_xlabel("1-Specificity")

    def auroc(self, lvl="m"):
        return trapz(self.roc[lvl][:, 0], self.roc[lvl][:, 1])

    def plot_comparison(
        self,
        uncertainty="lines",
        cims=None,
        ax=None,
        scatter_kws={},
        line_kws={},
        ellipse_kws={},
    ):
        ax = plt.gca() if ax is None else ax
        sns.scatterplot(
            data=self.all_scores,
            x="LR_Score_m",
            y="GP_Score_m",
            hue="Diagnosis",
            hue_order=["Viral", "Bacterial"],
            palette=self.palette,
            ax=ax,
            **({"legend": False} | scatter_kws),
        )

        ax.set_ylabel("Experimental Score")
        ax.set_xlabel("Statistical Score")

        cims = [1 / 4, 1 / 2, 3 / 4, 1] if cims is None else cims
        for dx, color in zip(["Viral", "Bacterial"], self.palette):
            for sample in self.all_scores[
                self.all_scores.Diagnosis == dx
            ].Sample.unique():
                these = self.all_scores[self.all_scores.Sample == sample]

                xy = (these.LR_Score_m.values, these.GP_Score_m.values)
                width = (these.LR_Score_u - these.LR_Score_l).values
                height = (these.GP_Score_u - these.GP_Score_l).values
                pair = np.array([-1, 1])

                if uncertainty == "lines":
                    ax.plot(
                        xy[0] + width / 2 * pair,
                        [xy[1], xy[1]],
                        color=color,
                        **line_kws,
                    )
                    ax.plot(
                        [xy[0], xy[0]],
                        xy[1] + height / 2 * pair,
                        color=color,
                        **line_kws,
                    )
                elif uncertainty == "ellipses":
                    for cim in cims:
                        ellipse = Ellipse(
                            xy=xy,
                            width=width * cim,
                            height=height * cim,
                            fc=color,
                            **({"alpha": 0.2} | ellipse_kws),
                        )
                        ax.add_patch(ellipse)

        x = np.array([-10, 10])
        reg = linregress(self.all_scores.LR_Score_m, self.all_scores.GP_Score_m)
        yl, xl = ax.get_ylim(), ax.get_xlim()
        ax.plot(x, self.reg.slope * x + self.reg.intercept, "k--")
        ax.set_ylim(yl)
        ax.set_xlim(xl)

