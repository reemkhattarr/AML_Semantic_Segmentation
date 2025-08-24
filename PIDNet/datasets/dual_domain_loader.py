import torch

class DualDomainLoader:
    def __init__(self, source_loader, target_loader):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.source_iter = iter(self.source_loader)
        self.target_iter = iter(self.target_loader)

    def __iter__(self):
        self.source_iter = iter(self.source_loader)
        self.target_iter = iter(self.target_loader)
        return self

    def __next__(self):
        try:
            source_batch = next(self.source_iter)
        except StopIteration:
            self.source_iter = iter(self.source_loader)
            source_batch = next(self.source_iter)
        try:
            target_batch = next(self.target_iter)
        except StopIteration:
            self.target_iter = iter(self.target_loader)
            target_batch = next(self.target_iter)
        return source_batch, target_batch
