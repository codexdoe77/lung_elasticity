import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import SimpleITK as sitk


def myshow(img, title=None, margin=0.05, dpi=80, img2=None, mask=True, colormap='gray', colorbar=False, colormax=20):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    
    
    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]
        
        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3,4):
            axnda = nda[nda.shape[0]//2,:,:]
            sagnda = nda[:,nda.shape[1]//2,:]
            cornda = nda[:,:,nda.shape[2]//2]
            
    # ysize = nda.shape[0]
    # xsize = nda.shape[1]
   
    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    # figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    # fig = plt.figure(figsize=figsize, dpi=dpi)
    fig, (axax, axsag, axcor) = plt.subplots(1,3)
    # Make the axis the right size...
    # ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
    # extent = (0, xsize*spacing[0], ysize*spacing[2], 0)
    
    tax = axax.imshow(axnda,colormap) #,extent=extent,interpolation=None)
    if nda.ndim == 2:
        tax.set_cmap(colormap)
    if colorbar:
        tax.set_clim(vmin=0, vmax=colormax)
        # plt.colorbar(t)

    tsag = axsag.imshow(sagnda,colormap) #,extent=extent,interpolation=None)
    if nda.ndim == 2:
        tsag.set_cmap(colormap)
    if colorbar:
        tsag.set_clim(vmin=0, vmax=colormax)

    tcor = axcor.imshow(cornda,colormap) #,extent=extent,interpolation=None)
    if nda.ndim == 2:
        tcor.set_cmap(colormap)
    if colorbar:
        tcor.set_clim(vmin=0, vmax=colormax)

    if (img2 is not None):
        nda2 = sitk.GetArrayFromImage(img2)
        axnda2 = nda2[nda2.shape[0]//2,:,:]
        sagnda2 = nda2[:,nda2.shape[1]//2,:]
        cornda2 = nda2[:,:,nda2.shape[2]//2]
        if mask:
            axax.imshow(axnda2,'jet',alpha=0.5*(axnda2>0)) # ,extent=extent,interpolation=None)
            axsag.imshow(sagnda2,'jet',alpha=0.5*(sagnda2>0))
            axcor.imshow(cornda2,'jet',alpha=0.5*(cornda2>0))
        else:
            axax.imshow(axnda2,'jet',alpha=0.5) # ,extent=extent,interpolation=None)
            axsag.imshow(sagnda2,'jet',alpha=0.5)
            axcor.imshow(cornda2,'jet',alpha=0.5)
        
    if(title):
        plt.title(title)

def pxshow(img, title=None, margin=0.05, dpi=80, img2=None, mask=True):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    
    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]
        
        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3,4):
            nda = nda[:,nda.shape[1]//2,:]
            
    ysize = nda.shape[0]
    xsize = nda.shape[1]
   
    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    fig = go.Figure(go.Heatmap(z=nda, colorscale='gray', showscale=False))

    if (img2 is not None):
        nda2 = sitk.GetArrayFromImage(img2)
        nda2 = nda2[:,nda2.shape[1]//2,:]
        fig.add_trace(go.Heatmap(z=nda2, colorscale='jet', opacity=0.5))

    fig.update_yaxes(scaleanchor='x', autorange='reversed', constrain='domain')
    fig.update_xaxes(constrain='domain')
    return fig