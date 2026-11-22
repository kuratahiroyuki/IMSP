import unittest
from data.generate_example import generate
from l1_lr_calibration_1 import prep_dataset, preselect_motor_and_sensory, DomainCorrelationReducer

class TestPreprocessing(unittest.TestCase):
    def test_representations(self):
        df = generate()
        df.columns = df.columns.str.replace('2 weeks::', '', regex=False).str.replace('6 months later::', '', regex=False)
        X, y = prep_dataset(df)
        self.assertEqual(X.shape[1], 132)
        self.assertEqual(set(y), {0, 1})
        for method in ('both', 'left', 'right', 'mean', 'smaller', 'larger'):
            with self.subTest(method=method):
                selected = preselect_motor_and_sensory(X, method)
                self.assertEqual(selected.shape[1], 132 if method == 'both' else 66)
                reducer = DomainCorrelationReducer().fit(selected)
                expected = 48 if method == 'both' else 38 if method in ('left', 'right') else 66
                self.assertEqual(selected.shape[1] - len(reducer.domain_removed_), expected)
                self.assertEqual(reducer.transform(selected).shape[1], len(reducer.selected_features_))
