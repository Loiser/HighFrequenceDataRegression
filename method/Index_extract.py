import torch

class Index_extract():

    def __init__(self,OEF,SP,N,K,J):
    
        self.N=N
        self.OEF=OEF.view(self.N,-1).cuda().detach()
        self.SP=SP.view(self.N,-1).cuda().detach()
        self.K=K
        self.J=J
        


    def get_beta(self):
    
        SP=self.SP
        OEF=self.OEF
        N=self.N
        K=self.K
        J=self.J
        
        b=K+J
        
        #求K_lag variatio
        SP_adK=torch.zeros(SP.shape,device=SP.device)
        SP_adK[0:N-K,:]=SP[K:N,:]
        SP_diffK=SP_adK-SP 

        OEF_adK=torch.zeros(OEF.shape,device=OEF.device)
        OEF_adK[0:N-K,:]=OEF[K:N,:]
        OEF_diffK=OEF_adK-OEF  

        cov1K=SP_diffK*OEF_diffK
        variat1K=(1/2)*cov1K[0:J].sum(0)+cov1K[J:N-b].sum(0)+(1/2)*cov1K[N-b:N-K].sum(0)

        cov2K=OEF_diffK*OEF_diffK
        variat2K=(1/2)*cov2K[0:J].sum(0)+cov2K[J:N-b].sum(0)+(1/2)*cov2K[N-b:N-K].sum(0)

        #求J_lag variation
        SP_adJ=torch.zeros(SP.shape,device=SP.device)
        SP_adJ[0:N-J,:]=SP[J:N,:]
        SP_diffJ=SP_adJ-SP

        OEF_adJ=torch.zeros(OEF.shape,device=OEF.device)
        OEF_adJ[0:N-J,:]=OEF[J:N,:]
        OEF_diffJ=OEF_adJ-OEF

        cov1J=SP_diffJ*OEF_diffJ
        variat1J=(1/2)*cov1J[0:K].sum(0)+cov1J[K:N-b].sum(0)+(1/2)*cov1J[N-b:N-J].sum(0)

        cov2J=OEF_diffJ*OEF_diffJ
        variat2J=(1/2)*cov2J[0:K].sum(0)+cov2J[K:N-b].sum(0)+(1/2)*cov2J[N-b:N-J].sum(0)

        #c
        c1=1/((1-b/N)*(K-J))*(variat1K-variat1J)
        c2=1/((1-b/N)*(K-J))*(variat2K-variat2J)

        # beta
        beta=c1*(1/c2)
        
        return(beta)
        
        
        
    def get_close(self):
        
         return(self.OEF[self.N -1],self.SP[self.N-1,:])
     
    def get_group(self):
        betas=Index_extract.get_beta(self)
        (_,sort_index)=torch.sort(betas)
            
        return(sort_index.view(10,-1))