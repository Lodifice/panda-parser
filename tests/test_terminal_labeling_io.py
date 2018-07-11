import io
import json
import unittest

import corpora.negra_parse as np
import grammar.induction.terminal_labeling as tl
import hybridtree.monadic_tokens as mt


class TestTerminalLabelingIO(unittest.TestCase):
    def test_simple_labelings(self):
        for labeling_class in [tl.PosTerminals, tl.FormTerminals, tl.CPosTerminals]:
            instance = labeling_class()
            serialization = instance.serialize()
            print(serialization)

            instance2 = tl.deserialize_labeling(serialization)

            self.assertTrue(isinstance(instance2, labeling_class))

    def test_composition_labeling(self):
        compositional = tl.CompositionalTerminalLabeling(tl.FormTerminals(), tl.PosTerminals(), binding_string='/')
        token = mt.ConstituentTerminal('Tisch', 'NN')
        label = compositional.token_label(token)
        print(token, label)

        serialization = compositional.serialize()
        print(serialization)

        instance2 = tl.deserialize_labeling(serialization)
        self.assertTrue(isinstance(instance2, compositional.__class__))
        self.assertEqual(label, instance2.token_label(token))

    def test_fallback_labeling(self):
        file = "res/TIGER/tiger21/tigertraindev_root_attach.export"
        corpus = np.sentence_names_to_hybridtrees([str(x) for x in range(50) if x % 10 > 1], file,
                                                  disconnect_punctuation=False)
        labeling = tl.FrequencyBiasedTerminalLabeling(tl.FormTerminals(), tl.PosTerminals(), corpus=corpus, threshold=2)
        print(labeling.fine_label_count)

        token1 = mt.ConstituentTerminal('Milliardär', 'NN')
        token2 = mt.ConstituentTerminal('Tisch', 'NN')
        label1 = labeling.token_label(token1)
        label2 = labeling.token_label(token2)

        f = io.StringIO()
        json.dump(labeling.serialize(), f)
        f.seek(0)
        print(f.getvalue())

        instance2 = tl.deserialize_labeling(json.load(f))

        self.assertTrue(isinstance(instance2, labeling.__class__))
        self.assertEqual(label1, instance2.token_label(token1))
        self.assertEqual(label2, instance2.token_label(token2))


if __name__ == '__main__':
    unittest.main()
