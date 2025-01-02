import unittest
import pandas as pd
import numpy as np
from main import replaceEmptyCellsWithNaN, formatCapitalizedCells



# Unit tests
class TestFunctions(unittest.TestCase):

    def test_replaceEmptyCellsWithNaN(self):
        # Test DataFrame with empty cells
        data = {
            'Column1': ['value1', '', 'value3'],
            'Column2': ['', 'value2', '']
        }
        df = pd.DataFrame(data)

        # Call the function
        replaceEmptyCellsWithNaN(df)

        # Expected result
        expected_data = {
            'Column1': ['value1', np.nan, 'value3'],
            'Column2': [np.nan, 'value2', np.nan]
        }
        expected_df = pd.DataFrame(expected_data)

        # Check if DataFrame matches expected result
        pd.testing.assert_frame_equal(df, expected_df)

    def test_formatCapitalizedCells(self):
        # Test cases for different inputs
        self.assertEqual(formatCapitalizedCells('HELLO'), 'Hello')  # All caps
        self.assertEqual(formatCapitalizedCells('Hello'), 'Hello')  # Already formatted
        self.assertEqual(formatCapitalizedCells('hello'), 'hello')  # Lowercase
        self.assertEqual(formatCapitalizedCells(123), 123)          # Non-string input
        self.assertEqual(formatCapitalizedCells(None), None)        # None input

if __name__ == '__main__':
    unittest.main()
