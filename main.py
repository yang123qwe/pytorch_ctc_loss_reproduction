import torch


class MY_CTC_LOSS(torch.nn.Module):
    def __init__(self ,blank = 0,reduction = 'none', ):
        super(MY_CTC_LOSS, self).__init__()
        self.blank = blank
        self.reduction = reduction
        
    def logadd(self,x0, x1, x2):
        return torch.logsumexp(torch.stack([x0, x1, x2]), dim = 0)


    def forward(self, log_probs : torch.Tensor, 
             targets : torch.Tensor, 
             input_lengths : torch.Tensor, 
             target_lengths : torch.Tensor, 
             finfo_min_fp32: float = torch.finfo(torch.float32).min,
             finfo_min_fp16: float = torch.finfo(torch.float16).min ):
        
        input_time_size, batch_size = log_probs.shape[:2]
        B = torch.arange(batch_size, device = input_lengths.device)

        targets_ = torch.cat([targets, targets[:, :1]], dim = -1)
        targets_ = torch.stack([torch.full_like(targets_, self.blank),
                                       targets_], dim = -1).flatten(start_dim = -2)

        diff_labels = torch.cat([torch.as_tensor([[False, False]], 
                                                 device = targets.device).expand(batch_size, -1),
                                 targets_[:, 2:] != targets_[:, :-2]], dim = 1)

        zero_padding, zero = 2, torch.tensor(finfo_min_fp16 
                                             if log_probs.dtype == torch.float16 else finfo_min_fp32, 
                                             device = log_probs.device, dtype = log_probs.dtype)
        log_probs_ = log_probs.gather(-1, targets_.expand(input_time_size, -1, -1))
        log_alpha = torch.full((1, batch_size, zero_padding + targets_.shape[-1]), 
                               zero, device = log_probs.device, dtype = log_probs.dtype)
        log_alpha[0, :, zero_padding + 0] = log_probs[0, :, self.blank]
        log_alpha[0, :, zero_padding + 1] = log_probs[0, B, targets_[:, 1]]
        zer_all = torch.full(( batch_size, 2 ),
                             zero, device = log_probs.device, 
                             dtype = log_probs.dtype)

        for t in range(1, input_time_size,1):
            temp = log_probs_[t] + self.logadd(log_alpha[t - 1, :, 2:], 
                                                         log_alpha[t - 1, :, 1:-1],
                                                         torch.where(diff_labels, 
                                                                     log_alpha[t - 1, :, :-2], zero))
            temp = torch.cat([zer_all ,temp ],-1).unsqueeze(0)
            log_alpha = torch.cat([log_alpha ,temp ],0)

        _loss = log_alpha[input_lengths - 1, 
                         B].gather(-1, torch.stack([zero_padding + target_lengths * 2 - 1, 
                                                    zero_padding + target_lengths * 2], dim = -1)) 
        loss = -torch.logsumexp(_loss, dim = -1)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.sum()/target_lengths.sum()
        else:
            raise ValueError("ValueError: {} is not a valid value for reduction".format(self.reduction))
def test_fun(device ,reduction_o):
    T, B, C = 300, 64, 128
    t = T // 2 - 4
    blank = 0
    
    logits = torch.randn(T, B, C, device = device).requires_grad_()
    targets = torch.randint(blank + 1, C, (B, t), dtype = torch.long, device = device)

    input_lengths = torch.full((B,), T, dtype = torch.long, device = device)
    target_lengths = torch.full((B,), t, dtype = torch.long, device = device)

    log_probs = logits.log_softmax(dim = -1)

    my_ctc_loss = MY_CTC_LOSS(blank = 0,reduction = reduction_o)
    torch_ctc_loss = torch.nn.CTCLoss(blank = 0,reduction = reduction_o)

    my_loss = my_ctc_loss(log_probs, targets, 
                          input_lengths, target_lengths)

    print ('device:{} ,reduction:{} , my_loss : {}'.format(device ,reduction_o , my_loss))
    torch_loss = torch_ctc_loss(log_probs, targets, 
                            input_lengths, target_lengths )
    print ('device:{} ,reduction:{} , torch_loss : {}'.format(device ,reduction_o ,torch_loss))
    
    print ('\n')
    return 
if __name__ == '__main__':
    for device in 'cpu','cuda':
        for reduction_o in 'none' , 'sum' , 'mean':
            test_fun(device,reduction_o)
    
