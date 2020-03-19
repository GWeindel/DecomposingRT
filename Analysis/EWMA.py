#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Copyright 2016 Michael D. Nunez

# This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 09/10/16      Michael Nunez                            Original code


# Exponentially weighted moving average
def ewmaseg(resps, rt, alpha=.05, limit=.6, plotit=True):
    """Determines rection time (RT) cutoffs using a basic exponentially weighted moving average
        of binomial responses sorted by RT.
    
    Let r denoted sorted binomial responses and s denote the exponentially weighted moving average
    Then all observations of s are given by the equations:
    s[0]==r[0]
    s[t]==alpha*r[t] + (1-alpha)*s[t-1]
    Similar to Joachim Vandekerckhove's ewmav2.m function in the DMAT toolbox.
    Parameters
    ----------
    resps : n array of 0's and 1's
    rt : n array of reaction times (any unit that is a multiple or factor of seconds)
    alpha: scalar paramter that weights the ewma to local values
    limit: cutoff accuracy decision threshold
    plotit: plot ewma and decision threshold
    """
    try:
        rt = np.squeeze(rt)
        resps = np.squeeze(resps)
        if (np.ndim(rt) != 1) or (np.ndim(resps) != 1):
            raise TypeError('rt and resps should be size n arrays')
        if rt.shape[0] != resps.shape[0]:
            raise ValueError('rt and resps should be size n arrays')
        # Sort data based on reaction time
        sortind = np.argsort(rt)
        sortrt = rt[sortind]
        sortresp = resps[sortind]

        # Calculate exponentially weighted moving average (ewma)
        N = rt.shape[0]
        s=np.empty(rt.shape[0])
        # s[0] = sortresp[0]
        s[0] = .5 # Force first value of ewma to .5
        for n in range(1,N):
            s[n] = alpha*sortresp[n] + (1-alpha)*s[n-1]
        
        # Find first crossing of the ewma across the limit boundary
        firstcross = np.where(s > limit)[0][0]
        rtbound = sortrt[firstcross]
        rm_mask = (rt < rtbound)

        # Plot results
        if plotit:
            plt.figure()
            # Plot properties
            teal = np.array([0, .7, .7])
            blue = np.array([0, 0, 1])
            orange = np.array([1, .3, 0])

            ewma = plt.plot(sortrt, s) #Exponentially Weighted Moving Average line
            plt.setp(ewma, color = teal, linewidth=2)
            axes = plt.gca()
            xmin, xmax = axes.get_xlim()
            ymin, ymax = axes.get_ylim()
            cutoff = plt.plot(np.array([xmin, xmax]),
                np.array([limit, limit])) #Cutoff line
            plt.setp(cutoff, color = blue, linewidth=2)
            minrt = plt.plot(np.array([rtbound, rtbound]),
                np.array([ymin, ymax])) #Suggested minimum reaction time
            plt.setp(minrt, color = orange, linewidth=2)  
            plt.ylabel('EWMA of Accuracy')
            plt.xlabel('Reaction time')
            plt.title('Exponentially Weighted Moving Average (alpha = %3.2f) of Sorted Accuracy' % (
                alpha))
    except (TypeError, ValueError) as error:
        print(error)
    return {'ewma':s, 'sortind':sortind, 'rm_mask':rm_mask, 'rtbound':rtbound }
