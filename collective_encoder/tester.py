import pytorch_lightning as pl

import collective_encoder.ce_run_base as crb

_SETTINGS = {
    'module': 'tester',
}

def test():
    """Test a collective encoder model based on the provided configuration."""

    config, metargs = crb.prepare(_SETTINGS)
    dm = crb.load_datamodule(config, metargs)
    model = crb.load_model(config, metargs, dm)
    run_dir = metargs['run_dir']

    ##################################
    testers = config.get('test_plotters', [])
    for tester in testers:
        model.add_test_plotter(tester['tester_type'], tester.get('tester_args', None))

    if isinstance(model, pl.LightningModule):
        trainargs = {"log_every_n_steps" : 1,
                     "default_root_dir" : run_dir}
        trainer = pl.Trainer(**trainargs)
        trainer.test(model, datamodule=dm)
    else:
        model.test(datamodule=dm)


def main():
    """Main entry point for the collective encoder testing."""
    test()

if __name__ == "__main__":
    main()





