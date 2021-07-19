import logging
import sys
import unittest

import pandas as pd

from solution import get_portfolio


class MainTestCase(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.df = pd.read_csv('resources/data.csv')

    @unittest.skip("until we'll have the full flow.")
    def test_main(self):
        portfolio = get_portfolio(self.df)
        assert True


if __name__ == '__main__':
    unittest.main()
