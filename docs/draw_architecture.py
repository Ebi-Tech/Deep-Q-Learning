"""Render Samuel's architecture diagram from the project's documented shapes."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(ax, x, y, label, color='#e5eefb', width=4.1, height=.72):
    ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,
                               boxstyle='round,pad=0.05',facecolor=color,
                               edgecolor='#314968',linewidth=1.2))
    ax.text(x,y,label,ha='center',va='center',fontsize=9)


def arrow(ax, start, finish, label=None):
    ax.annotate('',xy=finish,xytext=start,
                arrowprops=dict(arrowstyle='->',color='#314968',lw=1.5))
    if label:
        ax.text((start[0]+finish[0])/2+.1,(start[1]+finish[1])/2,label,
                fontsize=8,va='center')


fig, ax = plt.subplots(figsize=(13,10))
ax.set(xlim=(-.6,11),ylim=(-.7,10.6)); ax.axis('off')
ax.text(5.2,10.25,'Atari Pong — Deep Q-Network',ha='center',fontsize=20,weight='bold')
ax.text(2,9.6,'Shared online / target CNN',ha='center',fontsize=13,weight='bold')
left=[('4 stacked grayscale frames\n(B, 4, 84, 84), uint8 pixels',8.8),
      ('Normalize internally: float32 / 255',7.65),
      ('Conv 4→32, 8×8, stride 4 + ReLU\n(B, 32, 20, 20)',6.5),
      ('Conv 32→64, 4×4, stride 2 + ReLU\n(B, 64, 9, 9)',5.35),
      ('Conv 64→64, 3×3, stride 1 + ReLU\n(B, 64, 7, 7)',4.2),
      ('Flatten → 3136; Linear → 512 + ReLU',3.05),
      ('Linear 512 → A\nQ-values: (B, A); Pong A = 6',1.9)]
for label,y in left: box(ax,2,y,label)
for (_,a),(_,b) in zip(left,left[1:]): arrow(ax,(2,a-.4),(2,b+.4))
ax.text(7.6,9.6,'Interaction and training',ha='center',fontsize=13,weight='bold')
right=[('Pong → grayscale 84×84 → stack 4\nAction repetition: 4 emulator frames',8.8),
       ('Online CNN → epsilon-greedy action\nAct, store transition in replay',7.65),
       ('Replay minibatch: B transitions\nReward: sign(r); done: terminated only',6.5),
       ('Online Q(s,a) + target max Q(s′,a)\nTarget branch has no gradients',5.35),
       ('Bellman target: r + γ max Q(s′) (1−done)\nHuber loss → clip norm 10 → Adam',4.2),
       ('Hard-copy online → target\nEvery 1,000 environment decisions',3.05),
       ('Raw episode returns → CSV\nFinal weights → atomic checkpoint',1.9),
       ('play.py: greedy evaluation and MP4 capture\nSame preprocessing and CNN',.75)]
for label,y in right: box(ax,7.6,y,label,color='#e4f3ed',width=4.9)
for (_,a),(_,b) in zip(right,right[1:]): arrow(ax,(7.6,a-.4),(7.6,b+.4))
ax.text(2,.65,'B = batch size; A = action count\nCNN/preprocessing: Divine\nReplay/scheduling: Gabriel\nUpdates/logging/checkpoints/playback: Samuel',ha='center',va='center',fontsize=10)
ax.text(5.2,-.4,'Synchronization and saving are scheduled by the main loop; checkpoints contain inference weights.',ha='center',fontsize=9,color='#42556b')
fig.tight_layout()
folder=Path(__file__).resolve().parent
for extension in ('png','svg'):
    fig.savefig(folder/f'Samuel_architecture.{extension}',dpi=170,bbox_inches='tight')
