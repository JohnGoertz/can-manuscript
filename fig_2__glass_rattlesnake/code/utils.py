import pathlib as pl
import pandas as pd

code_pth = pl.Path(__file__).parent  # for running in terminal
fig_pth = code_pth.parent
data_pth = fig_pth / 'data'
graph_pth = fig_pth / 'graphics'
graph_pth.mkdir(exist_ok=True)
root_pth = fig_pth.parent

fig_version_path = root_pth / 'FIG_VERSION'
with fig_version_path.open('r') as file:
    version = file.read().strip()

with (root_pth / 'DIR_NAME_FMT').open('r') as file:
    dir_name_fmt = file.read().strip()

with (root_pth / 'FIG_NAME_FMT').open('r') as file:
    fig_name_fmt = file.read().strip()

alias_table = pd.read_csv(root_pth / 'figure_aliases.csv')
alias_to_number = alias_table.set_index('alias')[version].to_dict()
number_to_alias = alias_table.set_index(version)['alias'].to_dict()


def savefig(fig, alias, extra=''):
    number = alias_to_number[alias]
    name = fig_name_fmt.format(number=number, alias=alias, extra=extra)
    print(f"Saving {name} in {graph_pth}:", end="")
    for ext in ["svg", "png"]:
        try:
            fig.savefig(graph_pth / f"{name}.{ext}", dpi=600, transparent=True)
            print(f" [.{ext}]", end="")
        except AttributeError:
            print(f" [.{ext} failed]", end="")
    print(" done")
    