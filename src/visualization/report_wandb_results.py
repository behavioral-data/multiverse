import click
import wandb                         
api = wandb.Api()

from tabulate import tabulate
import pandas as pd

def get_all_runs(project):
    return api.runs(project)

def get_runs(project,names):
    if names:
        return get_runs_by_name(project,names)
    else:
        return get_all_runs(project)

def filter_runs(runs,fn):
    return [x for x in runs if fn(x)]

def get_runs_by_name(project,names,sort=True):
    runs = get_all_runs(project)
    filter_fn = lambda x: getattr(x,"name",None) in set(names)
    new_runs = filter_runs(runs, filter_fn)
    if sort:
        new_runs = sorted(new_runs, key =lambda x: names.index(x.name))
    return new_runs

def display_runs(runs,properties=None,fmt="ASCII"):
    results = []
    for run in runs:
        result = []
        result.append(run.name)
        result.append(run.notes)
        for p in properties:
            result.append(run.summary.get(p,None))
        results.append(result)
    
    columns = ["Name","Notes"] + list(properties)
    if fmt=="ASCII":
        print(tabulate(results, columns,floatfmt=".3f"))
    elif fmt=="latex":
        df = pd.DataFrame(results,columns=columns).round(3)
        print(df.to_latex(index=False))

@click.command()
@click.argument("project", type=str)
@click.option("-r","--run", "run_names", multiple=True, default=None, help="Run to report on (supports multuiple)")
@click.option("-p","--property", "properties", multiple=True, default=None, help="Properties to report (supports multiple)")
@click.option("-f","--format", "fmt", type=click.Choice(['ASCII', 'latex'], case_sensitive=False))
def main(project,
         run_names=None,
         properties=None,
         fmt="ASCII"):

    runs = get_runs(project,run_names)
    display_runs(runs,properties,fmt=fmt)
    

if __name__ == "__main__":
    main()
