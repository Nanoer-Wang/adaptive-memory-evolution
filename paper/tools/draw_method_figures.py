"""Vector illustrations of content, mutable state, and causal memory operations.

All glyphs are qualitative. The only plotted function is the analytic fixed-decay
reference, not experimental data. Geometry is specified at final paper width.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge, Polygon
from matplotlib.path import Path as MPath

P = Path(__file__).resolve().parents[1]
INK = '#193447'
MUTED = '#687B89'
LINE = '#CCD8DF'
TEAL = '#087F80'
BLUE = '#3D70AE'
AMBER = '#B47C2A'
PLUM = '#88709D'
STATE = [(TEAL, 'I', 'Importance'), (BLUE, 'C', 'Consistency'),
         (AMBER, 'U', 'Usage'), (PLUM, 'T', 'Age')]


def canvas(height):
    fig = plt.figure(figsize=(5.5, height), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 5.5), ylim=(0, height), aspect='equal')
    ax.axis('off')
    return fig, ax


def text(ax, x, y, value, size=8, color=INK, weight='normal', align='left', **kw):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                   ha=align, va='center', **kw)


def panel(ax, x, y, w, h, fill='#F6F9FA', edge=LINE, r=.035, lw=.65, **kw):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                      facecolor=fill, edgecolor=edge, linewidth=lw, **kw)
    ax.add_patch(p)
    return p


def line(ax, points, color=LINE, lw=.7, dashed=False, **kw):
    ax.plot(*zip(*points), color=color, lw=lw,
            linestyle=(0, (3, 2)) if dashed else '-', solid_capstyle='round', **kw)


def arrow(ax, a, b, color=MUTED, lw=.9, dashed=False, rad=0, scale=7, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=scale,
        linewidth=lw, color=color, linestyle=(0, (3, 2)) if dashed else '-',
        connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0, **kw))


def route(ax, points, color=MUTED, lw=.95, dashed=False):
    """Smooth cubic route with the arrowhead tangent to its last segment."""
    path = MPath(points, [MPath.MOVETO] + [MPath.CURVE4] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=7,
        color=color, linewidth=lw, linestyle=(0, (3, 2)) if dashed else '-'))


def dot(ax, x, y, radius=.024, color=TEAL, fill=None, lw=.75):
    ax.add_patch(Circle((x, y), radius, facecolor=fill or color,
                       edgecolor=color, linewidth=lw, zorder=4))


def document(ax, x, y, w=.22, h=.29, color=INK, fill='white', label=None, lw=.75):
    """Immutable content glyph, identical across all state snapshots."""
    x0, y0 = x - w / 2, y - h / 2
    fold = min(w, h) * .24
    ax.add_patch(Polygon([(x0, y0), (x0+w, y0), (x0+w, y0+h-fold),
                         (x0+w-fold, y0+h), (x0, y0+h)], closed=True,
                         facecolor=fill, edgecolor=color, linewidth=lw, joinstyle='round'))
    line(ax, [(x0+w-fold, y0+h), (x0+w-fold, y0+h-fold),
              (x0+w, y0+h-fold)], color, lw*.75)
    if label:
        text(ax, x, y-.02*h, label, size=8.2*(w/.22), align='center', color=color)
    else:
        for j, length in enumerate([.53, .53, .35]):
            yy = y0+h*(.62-j*.18)
            line(ax, [(x0+w*.19, yy), (x0+w*(.19+length), yy)], color, lw*.65)


def bubble(ax, x, y, w=.25, h=.20, color=BLUE, kind='query'):
    panel(ax, x-w/2, y-h/2, w, h, fill='white', edge=color, r=.04, lw=.85)
    ax.add_patch(Polygon([(x-w*.24,y-h/2+.005), (x-w*.30,y-h*.76),
                         (x-w*.02,y-h/2+.005)], facecolor='white', edgecolor=color, lw=.7))
    if kind == 'query':
        text(ax, x, y+.01, '?', size=10*(w/.25), weight='bold', align='center', color=color)
    else:
        for j in [-1, 0, 1]: dot(ax, x+j*.048, y, radius=.010, color=color, lw=.2)


def feedback_icon(ax, x, y, scale=1):
    dot(ax, x, y, .09*scale, AMBER, '#FFF8EC', lw=.7)
    text(ax,x,y+.002,r'$y$',8.7*scale,color=AMBER,align='center',zorder=5)


def memory_cell(ax, x, y, r=.32, strengths=(.80,.92,.40,.28), labels=False):
    """Four qualitative arcs: no normalized or calibrated state values implied."""
    ax.add_patch(Circle((x,y),r*.77,facecolor='#F6F9FA',edgecolor='#E2E9EC',lw=.45))
    sectors = [(96,174), (6,84), (186,264), (276,354)]
    for (color, symbol, _), (start,end), strength in zip(STATE,sectors,strengths):
        ax.add_patch(Wedge((x,y),r,start,end,width=r*.14,
                          facecolor=color,edgecolor='none',alpha=.14))
        ax.add_patch(Wedge((x,y),r,start,start+(end-start)*strength,
                          width=r*.14,facecolor=color,edgecolor='none'))
        if labels:
            theta = np.deg2rad((start+end)/2)
            text(ax,x+np.cos(theta)*r*1.22,y+np.sin(theta)*r*1.22,
                 symbol,7.3,color=color,weight='bold',align='center')
    document(ax,x,y,w=r*.62,h=r*.81,label=r'$x_i$',color=INK,lw=.8)


def label_heading(ax, x, y, letter, title):
    text(ax,x,y,letter,9.0,weight='bold')
    text(ax,x+.18,y,title,8.5,weight='bold')


def export(fig, name):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(P/'figures'/f'{name}.{ext}', dpi=360, facecolor='white',
                    metadata={'Creator': 'Adaptive Memory Evolution manuscript'})
    plt.close(fig)


def concept():
    fig, ax = canvas(1.76)
    label_heading(ax,.04,1.64,'a','Fixed decay')
    label_heading(ax,1.78,1.64,'b','Content persists; state evolves')
    line(ax,[(1.62,.17),(1.62,1.69)],'#DCE4E8',.7)

    document(ax,.27,1.08,.29,.39,label=r'$x_i$',color=INK)
    text(ax,.27,.75,'Content',6.7,color=MUTED,align='center')
    arrow(ax,(.47,1.11),(.67,1.11),color=MUTED,scale=6)
    origin=(.77,.82)
    arrow(ax,origin,(1.49,.82),color=MUTED,lw=.55,scale=4)
    arrow(ax,origin,(.77,1.30),color=MUTED,lw=.55,scale=4)
    ts=np.linspace(0,1,80)
    ax.plot(.80+.62*ts,.85+.40*np.exp(-3*ts),color=INK,lw=1.65)
    text(ax,1.12,1.41,r'$e^{-\lambda\,\mathrm{age}}$',9.0,align='center')
    text(ax,1.35,.71,'age',6.5,color=MUTED,align='center')
    text(ax,.80,.61,'Retention priority',6.7,color=MUTED,align='center')
    line(ax,[(.15,.41),(1.43,.41)],color=LINE,lw=.65)
    text(ax,.81,.23,'One prescribed decay path',7.1,align='center')

    snapshots=[(2.16,(.94,.92,.60,.25),r'$t_0$'),
               (3.57,(.26,.90,.30,.85),r'$t_1$'),
               (4.97,(.85,.92,.84,.16),r'$t_2$')]
    for x,strengths,time in snapshots:
        memory_cell(ax,x,1.02,r=.33,strengths=strengths,labels=True)
        text(ax,x,.54,time,8,color=MUTED,align='center')
    for x in [2.865,4.27]:
        arrow(ax,(x-.23,1.02),(x+.23,1.02),color=TEAL,lw=1.2,scale=7)
        text(ax,x,.86,r'$\Delta t$',7.3,color=MUTED,align='center')
    document(ax,2.80,1.27,w=.115,h=.155,color=TEAL,lw=.65)
    bubble(ax,2.97,1.27,w=.13,h=.12,color=BLUE)
    feedback_icon(ax,4.27,1.27,scale=.77)
    text(ax,2.87,1.46,'new context',6.5,color=MUTED,align='center')
    text(ax,4.27,1.46,'feedback',6.5,color=MUTED,align='center')
    for x,(color,symbol,label) in zip([1.84,2.88,4.02,4.83],STATE):
        dot(ax,x,.24,.028,color,lw=.2)
        text(ax,x+.075,.24,label,7.1,color=INK)
    # Crop only exterior whitespace; preserve every glyph's physical scale.
    fig.set_size_inches(5.5,1.56)
    ax.set_ylim(.16,1.72)
    export(fig,'concept')


def framework():
    fig, ax = canvas(2.52)
    for x,n,title in [(.04,'1','Experience'),(1.35,'2','Evolve state'),
                      (2.94,'3','Keep / release'),(4.55,'4','Read out')]:
        text(ax,x,2.40,n,8.6,color=MUTED,weight='bold')
        text(ax,x+.15,2.40,title,8.1,weight='bold')

    line(ax,[(.21,1.98),(.21,.81)],color='#D5E0E5',lw=3.3)
    for y,color in [(1.84,TEAL),(1.39,BLUE),(.94,AMBER)]:
        dot(ax,.21,y,.032,color,fill='white',lw=1.2)
    document(ax,.51,1.84,.20,.26,color=TEAL)
    bubble(ax,.51,1.39,.22,.19)
    feedback_icon(ax,.51,.94,.85)
    text(ax,.69,1.84,'observe',6.8,color=TEAL)
    text(ax,.69,1.39,'query',6.8,color=BLUE)
    text(ax,.69,.94,'feedback',6.8,color=AMBER)
    text(ax,.19,.66,'interleaved events',6.5,color=MUTED)
    arrow(ax,(1.04,1.40),(1.39,1.40),color=INK,lw=1.05)

    for j,(color,sym,pos) in enumerate([(TEAL,'I',.55),(BLUE,'C',.72),(AMBER,'U',.28)]):
        x=1.48+j*.40
        line(ax,[(x,2.12),(x+.27,2.12)],color='#CDD9DF',lw=1.5)
        dot(ax,x+.27*pos,2.12,.027,color,fill='white',lw=.85)
        text(ax,x+.135,2.25,rf'$\alpha_{sym}$',7.7,color=color,align='center')
    memory_cell(ax,1.94,1.47,r=.385,strengths=(.88,.81,.61,.36),labels=True)
    text(ax,1.94,.91,'flow  +  event updates',7.1,align='center')
    text(ax,1.94,.72,r'$\dot z_i=f(z_i,\mathrm{context})$',8.2,color=TEAL,align='center')

    panel(ax,3.20,.98,.99,1.10,fill='#F5F8FA',edge='#B7C9D3',r=.055,lw=.85)
    for row in range(2):
        for col in range(2):
            x=3.46+col*.46
            y=1.80-row*.47
            panel(ax,x-.175,y-.18,.35,.36,fill='white',edge='#D8E2E7',r=.022,lw=.5)
            document(ax,x-.018,y,w=.13,h=.18,color=INK,lw=.55)
            for j,(color,_,_) in enumerate(STATE):
                yy=y+.10-j*.064
                line(ax,[(x+.072,yy),(x+.124,yy)],color,lw=1.6)
    text(ax,3.70,.82,r'$K$ content slots',7.7,align='center')

    arrow(ax,(2.41,1.40),(2.69,1.40),color=TEAL,lw=1.1)
    document(ax,2.62,1.69,.14,.19,color=TEAL,lw=.65)
    text(ax,2.54,1.18,'write',6.7,color=TEAL,align='center')
    ax.add_patch(Polygon([(2.70,1.57),(2.97,1.57),(2.97,1.23),(2.70,1.23),
                          (2.77,1.40)],facecolor='#E8F3F1',edgecolor=TEAL,lw=.9,joinstyle='round'))
    text(ax,2.85,1.40,r'$S_i$',8.4,color=TEAL,align='center')
    arrow(ax,(2.98,1.40),(3.18,1.40),color=TEAL,lw=1.1)
    arrow(ax,(2.84,1.21),(2.84,.64),color=PLUM,lw=.85)
    route(ax,[(3.21,1.09),(3.02,1.09),(3.03,.76),(2.88,.64)],color=PLUM,lw=.85)
    document(ax,2.84,.50,.14,.19,color=PLUM,fill='#FBF9FC',lw=.65)
    text(ax,3.13,.55,'reject / evict',6.6,color=PLUM)
    text(ax,3.13,.38,r'overflow: lowest $S_i$',6.6,color=MUTED)

    route(ax,[(2.35,1.77),(2.59,2.28),(4.55,2.28),(4.86,1.91)],color=BLUE,lw=.95)
    text(ax,3.86,2.24,'query addressing',6.6,color=BLUE,align='center',
         bbox=dict(facecolor='white',edgecolor='none',pad=1))
    arrow(ax,(4.21,1.40),(4.48,1.40),color=INK,lw=1.05)
    document(ax,4.80,1.49,.27,.36,color=INK,lw=.8)
    ax.add_patch(Circle((4.84,1.60),.16,facecolor='#F3F7FC',edgecolor=BLUE,lw=1.1,alpha=.95))
    line(ax,[(4.96,1.49),(5.12,1.32)],BLUE,lw=2.0)
    line(ax,[(4.76,1.63),(4.91,1.63)],BLUE,lw=.7)
    line(ax,[(4.76,1.56),(4.88,1.56)],BLUE,lw=.7)
    arrow(ax,(4.83,1.23),(4.83,1.02),color=BLUE,lw=.85)
    bubble(ax,4.83,.89,w=.40,h=.23,color=BLUE,kind='answer')
    text(ax,4.83,.60,'answer',7.2,color=BLUE,align='center')

    route(ax,[(4.82,.42),(4.70,.10),(2.30,.08),(1.94,.48)],
          color=AMBER,lw=1.05,dashed=True)
    text(ax,3.22,.115,'delayed feedback  /  resident items only',6.8,color=AMBER,
         align='center',bbox=dict(facecolor='white',edgecolor='none',pad=1.0))
    fig.set_size_inches(5.5,2.40)
    ax.set_ylim(.07,2.47)
    export(fig,'framework')


def main():
    (P/'figures').mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,
                         'mathtext.fontset':'dejavusans','pdf.fonttype':42,
                         'svg.fonttype':'none'}):
        concept()
        framework()
    print('Redrew two illustrated mechanisms as vector PDF/SVG and 360-dpi PNG.')


if __name__=='__main__':
    main()
