import numpy as np
import torch
import tqdm


class GPUIndexer(object):
    def __init__(self, keys, gpus=(0,), fp16=False):
        self.gpus = gpus
        self.gpu = gpus[0]
        self.keys = keys
        self.fp16 = fp16
        self.dim = len(self.keys[0])

    def topk(self, query, topk: int = 1):
        raise NotImplementedError

    def batch_topk(self, query, topk: int = 1):
        raise NotImplementedError

    def batch_top1(self, query):
        raise NotImplementedError


class TorchGPUIndexer(GPUIndexer):
    def __init__(self, keys, gpus=(0,), fp16=False):
        super().__init__(keys, gpus, fp16)
        self.gpu_keys = torch.tensor(keys).cuda(self.gpu)
        print(f"Build torch indexer on GPU {self.gpu}")

        if self.fp16:
            self.gpu_keys = self.gpu_keys.half()

    def topk(self, query, topk: int = 1):
        if not type(query) is torch.Tensor:
            query = torch.tensor(query)
        query = query.cuda(self.gpu)
        if self.fp16:
            query = query.half()
        score = (self.gpu_keys * query).sum(-1)
        topk_score, topk_idx = score.topk(topk)
        return topk_score, topk_idx

    def batch_topk(self, query, topk: int = 1):
        if not type(query) is torch.Tensor:
            query = torch.tensor(query)
        query = query.cuda(self.gpu)
        if self.fp16:
            query = query.half()
        score = (self.gpu_keys.unsqueeze(0) * query.unsqueeze(1)).sum(-1)
        topk_score, topk_idx = score.topk(topk, dim=1)
        return topk_score, topk_idx

    def batch_top1(self, query):
        if not type(query) is torch.Tensor:
            query = torch.tensor(query)
        query = query.cuda(self.gpu)
        if self.fp16:
            query = query.half()
        score = (self.gpu_keys.unsqueeze(0) * query.unsqueeze(1)).sum(-1)
        topk_score, topk_idx = score.max(dim=1)
        return topk_score, topk_idx

    def batch_top1_l2(self, query):
        if not type(query) is torch.Tensor:
            query = torch.tensor(query)
        query = query.cuda(self.gpu)
        if self.fp16:
            query = query.half()
        # print(query.norm(dim=-1) - 1.)
        # print(self.gpu_keys.norm(dim=-1) - 1.)
        score = ((self.gpu_keys.unsqueeze(0) - query.unsqueeze(1)) ** 2).sum(-1)
        topk_score, topk_idx = score.min(dim=1)
        return topk_score, topk_idx


class FaissGPUIndexer(GPUIndexer):
    def __init__(self, keys, gpus=(0,), fp16=False):
        try:
            import faiss
        except Exception as e:
            print("Faiss is not installed! Please see https://github.com/facebookresearch/faiss/blob/master/INSTALL.md.")
            raise e
        super().__init__(keys, gpus, fp16)
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(self.dim)
        # index_flat = faiss.IndexFlatIP(self.dim)
        print(f"Build faiss indexer on GPU {self.gpu}")
        print(keys.shape)
        self.gpu_index_flat = faiss.index_cpu_to_gpu(res, self.gpu, index_flat)
        self.gpu_index_flat.add(keys)

    def batch_topk(self, query, topk: int = 1):
        if type(query) is torch.Tensor:
            query = query.cpu().numpy()
        D, I = self.gpu_index_flat.search(query, topk)
        D = D
        I = I
        D = torch.from_numpy(D)
        I = torch.from_numpy(I)
        return D, I

    def batch_top1(self, query):
        """
        :param query: shape of [b, f]
        """
        if type(query) is torch.Tensor:
            query = query.cpu().numpy()
        D, I = self.gpu_index_flat.search(query, 1)
        D = D[:, 0]
        I = I[:, 0]
        D = torch.from_numpy(D)
        I = torch.from_numpy(I)
        return D, I


if __name__ == '__main__':
    # 100k keys
    keys = np.random.uniform(size=(1000000, 64)) * 0.01
    querys = np.random.uniform(size=(1000000, 64)) * 0.01
    indexer = GPUIndexer(keys, [0], fp16=True)
    batch_size = 64
    for start in tqdm.tqdm(range(0, len(querys), batch_size)):
        query = querys[start: start + batch_size]
        # indexer.batch_topk(query, 1)
        top_score, top_idx = indexer.batch_top1(query)



