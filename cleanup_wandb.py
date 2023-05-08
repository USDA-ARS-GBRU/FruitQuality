import wandb

dry_run = False
project = wandb.Api().project('fruit-quality')

for artifact_type in project.artifacts_types():
    for artifact_collection in artifact_type.collections():
        for version in artifact_collection.versions():
            if artifact_type.type == 'model':
                if "latest" in version.aliases:
                    # print out the name of the one we are keeping
                    print(f'KEEPING {version.name}')
                else:
                    print(f'DELETING {version.name}')
                    if not dry_run:
                        print('')
                        version.delete(delete_aliases=True)
