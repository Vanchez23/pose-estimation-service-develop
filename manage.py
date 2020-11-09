# -*- coding: utf-8 -*-
import sys
import signal
import subprocess

import click

import settings
from src.server import Server


@click.group()
def cli():
    pass


@cli.command()
def run():
    server = Server()

    def signal_handler(signo, stack_frame):
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    server.run()


@cli.command()
@click.option('--coverage', is_flag=True)
def test(coverage):
    if not settings.TESTING_MODE:
        raise ValueError('You MUST set the variable TESTING_MODE == True')

    args = ['pytest', 'tests']
    if coverage:
        args.append('--cov=src')
    completed_process = subprocess.run(args)
    sys.exit(completed_process.returncode)


if __name__ == '__main__':
    cli()
