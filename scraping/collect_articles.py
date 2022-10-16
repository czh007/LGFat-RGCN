import argparse
import logging

from scraping import change_log_file_path, read_sources, scrape_and_store_articles

logger = logging.getLogger('scraping')


def main():
    parser = argparse.ArgumentParser(epilog=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'source_path',
        help='Path to a file containing a list of source URLs.'
    )
    parser.add_argument(
        'output_path',
        help='Path to the directory in which the articles should be stored.',
    )
    parser.add_argument(
        '--log-path',
        help='Path to log file (default: /tmp/scraping.log)',
        default='/tmp/scraping.log'
    )
    parser.add_argument(
        '-m', '--enable-mp',
        help='Enable multiprocessing (faster, but logging may be broken)',
        action='store_true'
    )
    parser.add_argument(
        '-n', '--num-workers',
        help='Number of concurrent worker processes/threads',
        type=int
    )
    args = parser.parse_args()

    source_path = args.source_path
    output_path = args.output_path
    log_path = args.log_path
    enable_mp = args.enable_mp
    num_workers = args.num_workers

    change_log_file_path(log_path)

    logger.info('Chinese in file...')
    article_sources = read_sources(source_path)
    if len(article_sources) == 0:
        raise RuntimeError(f'Aborting...')
    scrape_and_store_articles(
        article_sources,
        parent_dir=output_path,
        multiprocessing=enable_mp,
        num_workers=num_workers
    )


if __name__ == '__main__':
    main()
