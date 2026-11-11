import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('pipeline_worker',
    Path(__file__).resolve().parents[1]/'tools/pipeline_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerCatalogue(unittest.TestCase):
    def selected(self, names, first, layers):
        model = SimpleNamespace(tensors=[{'name': n} for n in names])
        return [t['name'] for t in worker.worker_tensors(model,first,layers)]

    def test_dense_suffix_excludes_embedded_draft(self):
        names = ['output.weight', 'output_norm.weight', 'token_embd.weight',
                 'blk.29.attn_norm.weight', 'blk.30.ffn_gate.weight',
                 'blk.63.attn_k.weight', 'blk.64.attn_k.weight',
                 'blk.64.nextn.eh_proj.weight']
        self.assertEqual(self.selected(names,30,64),
            ['output_norm.weight', 'blk.30.ffn_gate.weight', 'blk.63.attn_k.weight'])

    def test_moe_and_flash_next(self):
        self.assertEqual(self.selected(['output.scale', 'output_norm.weight',
            'blk.19.ffn_gate_exps.weight', 'blk.20.ffn_down_exps.weight',
            'blk.40.attn_q.weight'],20,40),
            ['output_norm.weight', 'blk.20.ffn_down_exps.weight'])
        self.assertEqual(self.selected(['token_embd.weight', 'output.weight',
            'output_hc_fn.weight', 'blk.23.attn_k.weight', 'blk.24.attn_k.weight'],24,48),
            ['output_hc_fn.weight', 'blk.24.attn_k.weight'])

    def test_mixed_weight_sizes(self):
        for kind, size in ((8,272), (13,176), (14,210), (20,144), (23,136)):
            self.assertEqual(worker.tensor_bytes({'kind':kind,'dims':(256,),'name':'weight'}),size)
        with self.assertRaises(ValueError):
            worker.tensor_bytes({'kind':14,'dims':(255,),'name':'partial'})


if __name__ == '__main__':
    unittest.main()
