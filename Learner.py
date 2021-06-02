
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    def update_net(self,loss, optimizer, net, net_share, scheduler):
        optimizer.zero_grad()
        if self.gpu:
            if self.args.alpha == 'auto':
                if net is not self.log_alpha:
                    net.zero_grad()
            else:
                net.zero_grad()
        loss.backward()
        if self.args.alpha == 'auto':
            if net is self.log_alpha:
                if self.log_alpha_share.grad is None or self.log_alpha_share.grad == 0:
                    self.log_alpha_share._grad = self.log_alpha.grad
            else:
                ensure_shared_grads(model=net, shared_model=net_share, gpu=self.gpu)
        else:
            ensure_shared_grads(model=net, shared_model=net_share, gpu=self.gpu)
        optimizer.step()
        scheduler.step(self.iteration)
    
    def target_q(self,r,done, q, q_std, q_next,log_prob_a_next):
        
        target_q = r + (1 - done) * self.args.gamma * (q_next - self.alpha.detach() * log_prob_a_next)

        if self.args.adaptive_bound:
            target_max = q + 3 * q_std
            target_min = q - 3 * q_std
            target_q = torch.min(target_q, target_max)
            target_q = torch.max(target_q, target_min)
        difference = torch.clamp(target_q - q, -self.args.TD_bound, self.args.TD_bound)
        target_q_bound = q + difference
        
        return target_q.detach(), target_q_bound.detach()
    
    def get_qloss(self,q, q_std, target_q, target_q_bound):
       
        if self.args.bound:
            loss = torch.mean(torch.pow(q - target_q, 2) / (2 * torch.pow(q_std.detach(), 2)) \
                                + torch.pow(q.detach() - target_q_bound, 2) / (2 * torch.pow(q_std, 2)) \
                                + torch.log(q_std))
        else:
            loss = -Normal(q, q_std).log_prob(target_q).mean()
        
        return loss

    q_1, q_std_1, _ = self.Q_net1.evaluate(s, a,device=self.device, min=False)
    a_next_1, log_prob_a_next_1, _ = self.actor1_target.evaluate(s_next, smooth_policy = smoothing_trick, device=self.device)
    q_next_1, _, q_next_sample_1 = self.Q_net1_target.evaluate(s_next, a_next_1, device=self.device, min=False)
    q_next_target_1 = q_next_sample_1
    target_q_1, target_q_1_bound = self.target_q(r, done, q_1.detach(), q_std_1.detach(), q_next_target_1.detach(), log_prob_a_next_1.detach())
    q_loss_1 = self.get_qloss(q_1, q_std_1, target_q_1, target_q_1_bound)
    self.update_net(q_loss_1, self.Q_net1_optimizer, self.Q_net1, self.Q_net1_share, self.scheduler_Q_net1)

          

