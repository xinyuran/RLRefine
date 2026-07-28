import unittest
from contextlib import redirect_stdout
from io import StringIO

from scripts.contract_quickstart import main


class ContractQuickstartTests(unittest.TestCase):
    def test_cpu_only_quickstart(self):
        output = StringIO()
        with redirect_stdout(output):
            status = main()
        self.assertEqual(status, 0)
        self.assertIn('"valid": true', output.getvalue())


if __name__ == "__main__":
    unittest.main()
