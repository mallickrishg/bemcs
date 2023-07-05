import numpy as np

def LDdispHS(x, y, xe, ye, a, dip, Ds, Dn, nu):
    # Arguments: (input)
    #   x & y  - The observation points locations in real Cartesian coords.  
    #  xe & ye - The element midpoint location in real coords. 
    #    a     - The elements half length
    #    dip   - dip angle in degrees
    #  Dn,Ds   - The defined displacement of each element.(normal and shear)
            #    Dn+ is opening, Ds+ is left lateral shearing. 
    #    nu    - The Poisson's ratio

    if np.any(y > 0):
        raise ValueError('Half-space solution: Z coordinates must be negative!')
    
    Beta = -np.deg2rad(dip)

    # Define material constant used in calculating influence coefficients.
    con = 1 / (4 * np.pi * (1 - nu))
    Dxb = Ds
    Dyb = -Dn
    sb = np.sin(Beta)
    cb = np.cos(Beta)
    s2b = np.sin(2 * Beta)
    c2b = np.cos(2 * Beta)
    s3b = np.sin(3 * Beta)
    c3b = np.cos(3 * Beta)
    
    # Define array of local coordinates for the observation grid relative 
    # to the midpoint and orientation of the ith element.
    XB = (x - xe) * cb + (y - ye) * sb
    YB = -(x - xe) * sb + (y - ye) * cb

    length = x.shape
    lengthrow = length[0]
    lengthcol = length[1]

    # Coordinates of the image dislocation
    XBi = (x - xe) * cb - (y + ye) * sb
    YBi = (x - xe) * sb + (y + ye) * cb
    # Fix roundoff errors
    bad = np.abs(YBi) < 1e-10
    bad2 = np.abs(YB) < 1e-10
    YBi[bad] = 0
    YB[bad2] = 0

    # Calculate derivatives of the function f(x,y), eq. 5.2.5 of C&S, p. 81.
    Y2 = YB ** 2
    XMa = XB - a
    XPa = XB + a
    XMa2 = XMa ** 2
    XPa2 = XPa ** 2
    R1S = XMa2 + Y2
    R2S = XPa2 + Y2

    Y2i = YBi ** 2
    XMai = XBi - a
    XPai = XBi + a
    XMa2i = XMai ** 2
    XPa2i = XPai ** 2
    R1Si = XMa2i + Y2i
    R1S2i = R1Si ** 2
    R2Si = XPa2i + Y2i
    R2S2i = R2Si ** 2

    # The following derivatives are eqs. 4.5.5a thru d of C&S, p. 58.
    FF2 = con * (np.log(np.sqrt(R1S)) - np.log(np.sqrt(R2S)))

    # Flag all observation points for which Yb is not 0
    i1 = np.where(YB)
    i2 = np.where((YB == 0) & (np.abs(XB) < a))
    i3 = np.where((YB == 0) & (np.abs(XB) > a))
    FF3 = np.zeros_like(YB)
    FF3[i1] = np.arctan2(YB[i1], XMa[i1]) - np.arctan2(YB[i1], XPa[i1])
    FF3[i2] = np.pi #* np.ones_like(i2)
    FF3[i3] = 0#np.zeros_like(i3)
    FF3 = -con * FF3.T
    FF3 = np.reshape(FF3, (lengthrow, lengthcol))

    FF4 = con * (YB / R1S - YB / R2S)
    FF5 = con * (XMa / R1S - XPa / R2S)

    FF2i = con * (np.log(np.sqrt(R1Si)) - np.log(np.sqrt(R2Si)))  # Equations 4.5.5 C&S
    # Flag all observation points for which Yb is not 0    
    i1i = np.nonzero(YBi)
    # Flag any observation points on the element
    i2i = np.nonzero(np.logical_and(YBi == 0, np.abs(XBi) < a))
    i3i = np.nonzero(np.logical_and(YBi == 0, np.abs(XBi) > a))
    # Steve Martels Solution to elements lying on same plane
    # FB3 = 0 for pts colinear with element, CON*pi for pts. on element 
    # FB3 = difference of arc tangents for all other pts.
    FF3i = np.zeros_like(YBi)
    FF3i[i1i] = np.arctan2(YBi[i1i], XMai[i1i]) - np.arctan2(YBi[i1i], XPai[i1i])
    FF3i[i2i] = np.pi #* np.ones_like(i2i)
    FF3i[i3i] = 0#np.zeros_like(i3i)
    FF3i = -con * FF3i.T
    FF3i = FF3i.reshape(lengthrow, lengthcol)
    FF4i = con * (YBi / R1Si - YBi / R2Si)
    FF5i = con * (XMai / R1Si - XPai / R2Si)

    FF6i = con * ((XMa2i - Y2i) / R1S2i - (XPa2i - Y2i) / R2S2i)
    FF7i = 2 * con * YBi * (XMai / R1S2i - XPai / R2S2i)

    # Define material constants used in calculating displacements.
    pr1 = 1 - 2 * nu
    pr2 = 2 * (1 - nu)
    pr3 = 3 - 4 * nu  # pr3 = 1 - pr
    # Calculate the displacement components using eqs. 5.5.4 of C&S, p. 91.
    Ux = Dxb * (-pr1 * sb * FF2 + pr2 * cb * FF3 + YB * (sb * FF4 - cb * FF5)) \
        + Dyb * (-pr1 * cb * FF2 - pr2 * sb * FF3 - YB * (cb * FF4 + sb * FF5))
    Uy = Dxb * (+pr1 * cb * FF2 + pr2 * sb * FF3 - YB * (cb * FF4 + sb * FF5)) \
        + Dyb * (-pr1 * sb * FF2 + pr2 * cb * FF3 - YB * (sb * FF4 - cb * FF5))

    # Calculate IMAGE AND SUPPLEMENTAL DISPLACEMENT components
    Uxi_s = Dxb * (pr1 * sb * FF2i - pr2 * cb * FF3i +
                (pr3 * (y * s2b - YB * sb) + 2 * y * s2b) * FF4i +
                (pr3 * (y * c2b - YB * cb) - y * (1 - 2 * c2b)) * FF5i +
                2 * y * (y * s3b - YB * s2b) * FF6i -
                2 * y * (y * c3b - YB * c2b) * FF7i)

    Uyi_s = Dxb * (-pr1 * cb * FF2i - pr2 * sb * FF3i -
                (pr3 * (y * c2b - YB * cb) + y * (1 - 2 * c2b)) * FF4i +
                (pr3 * (y * s2b - YB * sb) - 2 * y * s2b) * FF5i +
                2 * y * (y * c3b - YB * c2b) * FF6i +
                2 * y * (y * s3b - YB * s2b) * FF7i)

    # Calculate IMAGE AND SUPPLEMENTAL DISPLACEMENT components due to unit NORMAL
    Uxi_n = Dyb * (pr1 * cb * FF2i + pr2 * sb * FF3i -
                (pr3 * (y * c2b - YB * cb) - y) * FF4i +
                pr3 * (y * s2b - YB * sb) * FF5i -
                2 * y * (y * c3b - YB * c2b) * FF6i -
                2 * y * (y * s3b - YB * s2b) * FF7i)

    Uyi_n = Dyb * (pr1 * sb * FF2i - pr2 * cb * FF3i -
                pr3 * (y * s2b - YB * sb) * FF4i -
                (pr3 * (y * c2b - YB * cb) + y) * FF5i +
                2 * y * (y * s3b - YB * s2b) * FF6i -
                2 * y * (y * c3b - YB * c2b) * FF7i)

    Uxi = Uxi_s + Uxi_n
    Uyi = Uyi_s + Uyi_n

    Ux = Ux + Uxi
    Uy = Uy + Uyi

    return Ux,Uy

def LDstressHS(x, y, xe, ye, a, dip, Ds, Dn, nu, mu):
    # Arguments: (input)
    #   x & y  - The observation points locations in real Cartesian coords.  
    #  xe & ye - The element midpoint location in real coords. 
    #    a     - The elements half length
    #    dip   - dip angle in degrees
    #  Dn,Ds   - The defined displacement of each element.(normal and shear)
            #    Dn+ is opening, Ds+ is left lateral shearing. 
    #    nu    - The Poisson's ratio
    #    mu    - Shear Modulus (E     - The Young's modulus E = 2*G*nu/(1-nu))
    #  Arguments: (output)
    #  Stress - Is the stress caused by the movement of the dislocation at the observataion points. 
    # [Sxx,Syy,Sxy]

    Beta = -np.deg2rad(dip)

    if np.any(y > 0):
        raise ValueError('Half-space solution: Z coordinates must be negative!')
    
    # The shear modulus, sm, is related to the prescribed elastic constants.
    # sm = E / (2 * (1 + nu))
    sm = mu
    # Define material constant used in calculating influence coefficients.
    con = 1 / (4 * np.pi * (1 - nu))
    cons = 2 * sm
    Dxb = Ds
    Dyb = -Dn
    sb = np.sin(Beta)
    cb = np.cos(Beta)
    s2b = np.sin(2 * Beta)
    c2b = np.cos(2 * Beta)
    s3b = np.sin(3 * Beta)
    c3b = np.cos(3 * Beta)
    s4b = np.sin(4 * Beta)
    c4b = np.cos(4 * Beta)
    
    # Define array of local coordinates for the observation grid relative to
    # the midpoint and orientation of the ith element.
    # Refer to (Figure 5.6, C&S, p. 91) and eqs. 4.5.1 of C&S, p. 57.
    XB = (x - xe) * cb + (y - ye) * sb
    YB = -(x - xe) * sb + (y - ye) * cb 
    
    # Coordinates of the image dislocation
    XBi = (x - xe) * cb - (y + ye) * sb  # equation 7.4.6 C&S
    YBi = (x - xe) * sb + (y + ye) * cb
    
    # Fix roundoff errors in Ybi and Yb from trig function problems
    bad = np.where(np.abs(YBi) < 1e-10)  # Steve Martels Fix
    bad2 = np.where(np.abs(YB) < 1e-10)
    YBi[bad] = 0
    YB[bad2] = 0.
    
    # Calculate derivatives of the function f(x, y), eq. 5.2.5 of C&S, p. 81.
    # which are used to calculate the displacement and stress components.
    # It is understood that X and Y refer to XB and YB.
    # First abbreviate repeated terms in the derivatives of f(x, y):
    Y2 = YB ** 2
    XMa = XB - a
    XPa = XB + a
    XMa2 = XMa ** 2
    XPa2 = XPa ** 2
    R1S = XMa2 + Y2
    R1S2 = R1S ** 2
    R2S = XPa2 + Y2
    R2S2 = R2S ** 2
    
    # Same thing for the image dislocation
    Y2i = YBi ** 2
    XMai = XBi - a
    XPai = XBi + a
    XMa2i = XMai ** 2
    XPa2i = XPai ** 2
    R1Si = XMa2i + Y2i
    R1S2i = R1Si ** 2
    R2Si = XPa2i + Y2i
    R2S2i = R2Si ** 2
    
    FF4 = con * (YB / R1S - YB / R2S)
    FF5 = con * (XMa / R1S - XPa / R2S)
    # The following derivatives are eqs. 5.5.3a and b of C&S, p. 91.
    FF6 = con * ((XMa2 - Y2) / R1S2 - (XPa2 - Y2) / R2S2)
    FF7 = 2 * con * YB * (XMa / R1S2 - XPa / R2S2)
    
    FF4i = con * (YBi / R1Si - YBi / R2Si)
    FF5i = con * (XMai / R1Si - XPai / R2Si)
    
    # The halfspace examples of eqs. 5.5.3a and b of C&S, p. 91.
    # See Appendix A of: Martel, S.J. and Langley, J.S., 2006. Propagation of
    # normal faults to the surface in basalt, Koae fault system, Hawaii.
    # Journal of Structural Geology, 28(12), pp.2123-2143.
    FF6i = con * ((XMa2i - Y2i) / R1S2i - (XPa2i - Y2i) / R2S2i)
    FF7i = 2 * con * YBi * (XMai / R1S2i - XPai / R2S2i)
    
    # *Tim* I used MATLAB's symbolic to find these not eq's A.3 and A.4 of Martel
    # Used Eq.A.1 on variable FF7i (expanded).
    FF8i = (YBi * (1 / ((a + XBi) ** 2 + YBi ** 2) ** 2 - 1 / (YBi ** 2 + (a - XBi) ** 2) ** 2 +
                    (2 * (a - XBi) * (2 * a - 2 * XBi)) / (YBi ** 2 + (a - XBi) ** 2) ** 3 -
                    (2 * (a + XBi) * (2 * a + 2 * XBi)) / ((a + XBi) ** 2 + YBi ** 2) ** 3)) / (2 * np.pi * (nu - 1))
    FF9i = ((a - XBi) / (YBi ** 2 + (a - XBi) ** 2) ** 2 + (a + XBi) / ((a + XBi) ** 2 + YBi ** 2) ** 2) / (2 * np.pi * (nu - 1)) - \
           (YBi * ((4 * YBi * (a + XBi)) / ((a + XBi) ** 2 + YBi ** 2) ** 3 +
                   (4 * YBi * (a - XBi)) / (YBi ** 2 + (a - XBi) ** 2) ** 3)) / (2 * np.pi * (nu - 1))
    
    # Calculate the stress components using eqs. 5.5.5 of C&S, p. 92.
    Sxx = cons * Dxb * (2 * (cb * cb) * FF4 + s2b * FF5 + YB * (c2b * FF6 - s2b * FF7)) + cons * Dyb * (
            -FF5 + YB * (s2b * FF6 + c2b * FF7))
    Syy = cons * Dxb * (2 * (sb * sb) * FF4 - s2b * FF5 - YB * (c2b * FF6 - s2b * FF7)) + cons * Dyb * (
            -FF5 - YB * (s2b * FF6 + c2b * FF7))
    Sxy = cons * Dxb * (s2b * FF4 - c2b * FF5 + YB * (s2b * FF6 + c2b * FF7)) + cons * Dyb * (
            -YB * (c2b * FF6 - s2b * FF7))
    
    # Calculate IMAGE AND SUPPLEMENTAL STRESS components due to unit SHEAR and
    # NORMAL displacement discontinuity
    Sxxi_s = cons * Dxb * (FF4i - 3 * (c2b * FF4i - s2b * FF5i) + (2 * y * (cb - 3 * c3b) + 3 * YB * c2b) * FF6i +
                           (2 * y * (sb - 3 * s3b) + 3 * YB * s2b) * FF7i -
                           2 * y * (y * c4b - YB * c3b) * FF8i - 2 * y * (y * s4b - YB * s3b) * FF9i)
    Sxxi_n = cons * Dyb * (FF5i + (2 * y * (sb - 2 * s3b) + 3 * YB * s2b) * FF6i -
                           (2 * y * (cb - 2 * c3b) + 3 * YB * c2b) * FF7i - 2 * y * (y * s4b - YB * s3b) * FF8i +
                           2 * y * (y * c4b - YB * c3b) * FF9i)
    
    Syyi_s = cons * Dxb * (FF4i - (c2b * FF4i - s2b * FF5i) -
                           (4 * y * sb * s2b - YB * c2b) * FF6i +
                           (4 * y * sb * c2b + YB * s2b) * FF7i +
                           2 * y * (y * c4b - YB * c3b) * FF8i +
                           2 * y * (y * s4b - YB * s3b) * FF9i)
    Syyi_n = cons * Dyb * (FF5i - (2 * y * sb - YB * s2b) * FF6i +
                           (2 * y * cb - YB * c2b) * FF7i +
                           2 * y * (y * s4b - YB * s3b) * FF8i -
                           2 * y * (y * c4b - YB * c3b) * FF9i)
    
    Sxyi_s = cons * Dxb * (s2b * FF4i + c2b * FF5i +
                           (2 * y * sb * (1 + 2 * s2b) + YB * c2b) * FF6i -
                           (2 * y * sb * (1 - 2 * c2b) + YB * s2b) * FF7i -
                           2 * y * (y * s4b - YB * s3b) * FF8i -
                           2 * y * (y * c4b - YB * c3b) * FF9i)
    Sxyi_n = cons * Dyb * (YB * FF5i + (2 * y * sb + YB * s2b) * FF6i -
                           (2 * y * cb + YB * c2b) * FF7i -
                           2 * y * (y * c4b - YB * c3b) * FF8i +
                           2 * y * (y * s4b - YB * s3b) * FF9i)
    
    Sxxi = Sxxi_s+Sxxi_n
    Syyi = Syyi_s+Syyi_n
    Sxyi = Sxyi_s+Sxyi_n
    Sxx = Sxx+Sxxi
    Syy = Syy+Syyi
    Sxy = Sxy+Sxyi

    return Sxx,Syy,Sxy