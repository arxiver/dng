
from .blocks import *
def Discriminator(opt):
    input_nc = opt.input_nc
    ndf = opt.ndf
    netD_n_layers=opt.netD_n_layers 
    n_domains=opt.n_domains
    tensor = opt.Tensor
    norm = opt.norm or 'batch'
    gpu_ids=opt.gpu_ids
    norm_layer = get_norm_layer(norm_type=norm)
    model_args = (input_nc, ndf, netD_n_layers, tensor, norm_layer, gpu_ids)
    plex_netD = D_Plexer(n_domains, NLayerDiscriminator, model_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netD.cuda(gpu_ids[0])

    plex_netD.apply(weights_init)
    return plex_netD
