import fs from 'node:fs/promises';

const ROOT='D:/program/LEO_switch';
const BUILD=ROOT+'/tmp/pptx_report_20261030';
const C={ink:'#16324F',blue:'#2463A6',teal:'#117D83',muted:'#5B6877',light:'#EDF3F8',line:'#D5DFE8',white:'#FFFFFF'};
const FONT='Microsoft YaHei';
const txt=(s,t,x,y,w,h,size=26,color=C.ink,bold=false,align='left')=>{
 const a=s.shapes.add({geometry:'textbox',name:t.slice(0,35),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=t; a.text.style={fontSize:size,typeface:FONT,color,bold,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'square',insets:0};return a;
};
function table(s,values,x,y,w,h,widths,size=24){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 t.borders.assign({fill:C.line,width:1,style:'solid'});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const a=t.getCell(r,c);a.fill=r===0?C.ink:(r%2?C.white:C.light);
  a.text.style={typeface:FONT,fontSize:size,bold:r===0,color:r===0?C.white:C.ink,verticalAlignment:'middle',insets:12};
 }
 return t;
}
async function pic(s,n,x,y,w,h){
 return s.images.add({path:BUILD+'/assets/'+n+'.png',contentType:'image/png',alt:n,fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
function base(p,title,n,notes){
 const s=p.slides.add({width:1280,height:720});s.background.fill=C.white;
 txt(s,title,56,35,1168,64,43,C.ink,true);
 txt(s,String(n).padStart(2,'0'),1174,672,50,25,17,C.muted,false,'right');
 s.speakerNotes.text=notes;return s;
}
const provenance='数据来源：D:/program/LEO_switch/results/multiseed_0907/multiuser_summary.csv 和 multiuser_seed_records.csv；使用断电异常修复后的有效结果。训练种子42–46，先汇总每种子的5个评估回合，再对5训练种子汇总。';

export async function build(p){
 p.slides.keep([]);
 // 1: cover and background combined, to keep all content within ten slides.
 let s=base(p,'',1,'背景：地面覆盖受限区域的终端需要远端计算支持，LEO卫星搭载MEC可扩展服务范围。终端算力有限，但卫星持续运动、带宽与计算资源共享，选择哪颗星和卸载多少任务必须一起考虑。本文采用部分卸载，本地与远端并行。图源：paper/系统架构图.drawio.pdf。论文来源：C:/Users/19704/Desktop/leo_switch.pdf，引言与系统模型。');
 txt(s,'LEO 卫星边缘计算',56,62,700,77,54,C.ink,true);
 txt(s,'联合切换与任务卸载',56,145,710,77,54,C.ink,true);
 txt(s,'HAN+PDQN',60,250,630,55,39,C.blue,true);
 txt(s,'面向动态拓扑与多用户资源竞争',60,312,650,48,28,C.muted);
 txt(s,'背景与动机',60,420,630,40,27,C.ink,true);
 txt(s,'远端计算补足终端算力\n卫星运动改变服务机会\n共享资源使“选星”与“卸载比例”相互影响',60,470,655,150,27,C.muted);
 await pic(s,'system',760,170,464,464);
 txt(s,'项目汇报  ·  2026.10',60,672,600,25,17,C.muted);

 // 2: research gap + challenges.
 s=base(p,'现有研究与本项目切入点',2,'现有工作已研究图学习辅助切换、联合接入与卸载、多智能体控制及参数化混合动作，不能概括成“尚无人研究联合优化”。本文更具体的切入点是：已有与潜在竞争如何同时影响每个候选卫星的卸载请求和选择价值。参考现有稿件引言：[3] Shen et al., JSAC 2021, Graph Neural Networks for Scalable Radio Resource Management；[4] Fu et al., Chinese Journal of Computers 2026, A Multi-Agent Heterogeneous Graph Representation-Based Handover Method；[5] Yang et al., TCCN 2025, Joint Access Selection and Computation Offloading；[10] Xiong et al., arXiv:1810.06394, 2018；[11] Yang et al., Sensors 2023, Computation Offloading and Resource Allocation Based on P-DQN. 来源：C:/Users/19704/Desktop/leo_switch.pdf，第I节。');
 table(s,[['研究方向','已有能力','本问题仍需补足'],['图学习与卫星切换','表达网络关系、辅助选星','与候选卫星绑定的卸载比例'],['联合卸载与资源分配','处理接入、计算和动态任务','显式区分当前与潜在竞争'],['混合动作强化学习','离散选择 + 连续控制','把竞争表示注入比例与价值']],56,134,1168,300,[285,390,493],25);
 txt(s,'挑战 1',56,476,180,40,27,C.blue,true);
 txt(s,'动态竞争关系\n谁正在共享，谁可能迁入',56,525,352,80,26);
 txt(s,'挑战 2',457,476,180,40,27,C.blue,true);
 txt(s,'离散–连续耦合\n选哪颗星，卸载多少',457,525,352,80,26);
 txt(s,'挑战 3',858,476,180,40,27,C.blue,true);
 txt(s,'跨时隙影响\n队列、残余任务与切换开销',858,525,366,80,26);
 txt(s,'研究切入点：让同一竞争表示同时影响卸载比例和卫星选择价值',56,633,1120,35,25,C.teal,true);

 // 3: original framework and a precise implementation label.
 s=base(p,'HAN+PDQN 总体框架',3,'按图从左到右讲解：环境生成用户、卫星、连接、可见性、任务和资源状态；HAN形成用户关系表示；与局部任务及候选卫星信息融合；共享PDQN参数网络给出候选卸载比例，Q网络评价卫星–比例对，经可行性掩码选择联合动作；环境按迁移、接纳和本地回退规则执行。当前实现先预训练并冻结HAN，再通过编码观测的经验回放训练PDQN，不是端到端联合训练。图中lambda表示连续卸载比例，与本文rho同义。原PDF保留；源图英文标签Heterogeous为Heterogeneous的拼写错误。来源：paper/算法框架图10_23.pdf；scripts/train.py::HANPDQNTrainer；src/algorithm/pdqn.py。');
 await pic(s,'framework',56,125,1168,526);
 txt(s,'关系编码 → 候选比例 → 联合价值 → 可行动作',56,99,1168,28,24,C.blue,true);
 txt(s,'训练机制：预训练并冻结 HAN；PDQN 使用经验回放与目标网络',56,660,1100,32,23,C.teal,true);

 // 4: editable semantic table instead of another dense diagram.
 s=base(p,'HAN：区分已有竞争与潜在竞争',4,'U代表用户，S代表卫星。Serving U-S-U表示当前由同一卫星服务的用户，其带宽与MEC资源已形成共享关系；Visibility U-S-U连接具有共同候选卫星的用户，表达未来可能汇聚的需求，不能理解为它们必然同时迁入。用户侧另有近邻路径U-U；卫星侧有可见路径S-U-S和星间连接路径S-S-S。实现采用带边特征的逐跳节点级注意力，再按节点类型做语义级融合；同一图的同类节点共享语义权重。多层传播后形成用户嵌入，与原始任务及候选观测拼接。公式为汇报简写，不额外声称注意力权重具备因果解释。来源：论文IV-B、Table I；src/model/hetero_gnn.py。');
 txt(s,'用户 U  /  卫星 S',56,124,650,44,28,C.blue,true);
 table(s,[['关系路径','表达的竞争 / 依赖'],['U–S–U：当前服务','已有资源共享'],['U–S–U：共同可见','潜在接入竞争'],['U–U：地理近邻','邻近用户需求'],['S–U–S / S–S–S','重叠服务机会 / 星间拓扑']],56,185,680,342,[306,374],24);
 txt(s,'节点级注意力',790,147,428,42,30,C.blue,true);
 txt(s,'同一路径内\n聚合不同邻居的贡献',790,203,428,83,27);
 txt(s,'语义级注意力',790,328,428,42,30,C.blue,true);
 txt(s,'融合不同关系路径\n形成用户嵌入 zᵢ',790,384,428,83,27);
 txt(s,'PDQN 输入：用户嵌入 + 本地任务 + 候选卫星状态',56,575,1168,62,30,C.teal,true);

 // 5: core forward rule plus actual training loop.
 s=base(p,'PDQN：为每颗候选卫星匹配卸载比例',5,'离散动作包括stay及各候选卫星槽位。参数网络依据相同融合输入x_i生成每个离散动作专属的连续参数rho_i,k，Q网络评价(x_i,k,rho_i,k)配对输入。动作掩码剔除当前不满足连接和切换条件的候选，环境提交时再执行容量接纳与回退。所有用户共享网络权重。Q学习使用在线网络选择后继动作、目标网络评价的TD目标；参数网络提升掩码后的最大Q值，目标网络周期复制。经验池保存编码观测、请求动作、奖励、后继编码观测、终止标志和前后掩码。HAN被冻结。奖励综合任务成败、时延、终端能耗、中断、切换和资源公平性；不是仅优化成功条件下时延。来源：论文IV-C/D，src/algorithm/pdqn.py。');
 txt(s,'在线决策',56,127,550,45,31,C.blue,true);
 txt(s,'① 生成每个候选的连续比例',56,201,690,42,28,C.ink,true);
 txt(s,'ρᵢ,ₖ = μₖ(xᵢ)  ∈ [0, 1]',94,254,670,56,34);
 txt(s,'② 在可行候选中选择最大 Q 值',56,354,690,42,28,C.ink,true);
 txt(s,'k* = arg maxₖ Q(xᵢ, k, ρᵢ,ₖ)',94,407,680,57,32);
 txt(s,'③ 执行“卫星 k* + 对应比例 ρᵢ,ₖ*”',56,507,720,44,28,C.teal,true);
 txt(s,'训练更新',830,127,394,45,31,C.blue,true);
 txt(s,'环境反馈\n编码观测经验池\nQ 网络：拟合 TD 目标\n参数网络：提高 Q 值\n目标网络：周期复制',830,203,394,292,26);
 txt(s,'长期回报兼顾：任务成败 · 时延 · 终端能耗 · 服务连续性',56,604,1168,55,27,C.ink,true);

 // 6: experimental setup, grouped baseline descriptions.
 s=base(p,'实验设置与对比基线',6,'正式主对比共7种方法，即HAN+PDQN和6个基线。主方法及各学习方法分别在20、25、30、35、40用户规模训练，不能称为跨规模泛化。66星、6轨道面、高度550km、倾角53度，本地处理率1 Gcycle/s，MEC两核总处理率10 Gcycle/s。每种子150k交互步，每回合512步；训练种子42–46，每种子5评估回合，均值和置信区间按训练种子汇总。按评估回报选取checkpoint，当前场景与模型选择评估有复用，独立测试作为后续工作。主方法和无图PDQN的replay batch分别512/128，完整方法比较不等同于只改变HAN的受控消融。Min-Distance用最近可行卫星和固定全卸载请求，并受统一接纳和本地回退规则约束。SCA已有seed42单种子配对补充，复用HAN+PDQN离散决策，不在本页五种子主比较中。来源：论文V-A，paper/README.md，results/multiseed_0907。');
 txt(s,'66 星  ·  550 km  ·  20–40 用户',56,132,1168,48,35,C.blue,true);
 txt(s,'5 个训练种子 × 每种子 5 回合评估  |  150k 交互步  |  95% bootstrap 区间',56,191,1168,43,25,C.muted);
 table(s,[['对比方法','比较目的'],['PDQN','不使用 HAN 的参数化混合动作方法'],['HAN+MAPPO / MAPPO','图表示 + 多智能体策略优化 / 无图版本'],['MADDPG','多智能体 actor–critic 对照'],['Min-Distance / Full-Local','最近卫星 + 全卸载请求 / 全本地执行']],56,270,1168,305,[425,743],25);
 txt(s,'主方法默认：学习率 10⁻³，replay batch = 512',56,598,1168,37,25,C.teal,true);
 txt(s,'PDQN 基线 batch = 128；当前比较评价完整方法，严格受控消融待补',56,646,1120,32,22,C.muted);

 // 7: both source convergence PDFs, with separate readable layouts.
 s=base(p,'收敛表现与参数敏感性（30 用户）',7,'左图为5种学习算法的训练回报：HAN+PDQN后期平均回报最高，约100回合后进入较高回报区间；MAPPO类早期上升更快，因此不宣称主方法训练最快。右侧分别呈现学习率、replay batch size的敏感性。学习率3e-3的均值偏低且种子差异较大；batch=256均值偏低；1e-3/512为表现良好的默认组合，不能声称唯一最优或batch越大越好。算法对比图未平滑；超参数图使用3回合向后移动平均。均为5训练种子的95% bootstrap区间。超参数共7种唯一配置、每组5种子。来源：paper/algorithm_convergence_u30.pdf、paper/convergence_u30.pdf及对应figure/paper_candidate manifests。');
 txt(s,'主方法后期平均回报较高',56,123,670,42,28,C.blue,true);
 await pic(s,'algorithms',44,183,725,434);
 await pic(s,'parameters',803,122,414,538);
 txt(s,'默认参数：10⁻³ / 512；更大学习率或 batch 不保证更优',56,638,720,45,22,C.muted);

 // 8: full-width original task result figure and three succinct numeric takeaways.
 s=base(p,'任务性能：违约率相对 PDQN 降低 21.26%',8,provenance+'图源：paper/figure1_line_metrics.pdf。三个相对下降先对20/25/30/35/40用户的指标均值取算术平均，再计算1-主方法/PDQN。成功任务时延1.750827s vs 1.789413s，下降2.156%；截止违约率7.09934% vs 9.01610%，相对下降21.259%（绝对下降1.91676个百分点）；每成功任务能耗1.230932J vs 1.305653J，下降5.72285%。时延仅针对成功任务，必须与违约率结合解读；每成功任务能耗是全部记账终端能耗除以成功数。图中置信区间不自动构成差值显著性检验。');
 await pic(s,'task_metrics',42,120,1196,454);
 txt(s,'成功任务平均时延',56,584,365,35,25,C.muted);
 txt(s,'↓ 2.16%',56,623,365,50,39,C.blue,true);
 txt(s,'截止时间违约率',457,584,365,35,25,C.muted);
 txt(s,'↓ 21.26%',457,623,365,50,39,C.blue,true);
 txt(s,'每成功任务能耗',858,584,365,35,25,C.muted);
 txt(s,'↓ 5.72%',858,623,365,50,39,C.blue,true);
 txt(s,'相对 PDQN；先取五规模均值，再计算相对变化',56,686,1080,22,16,C.muted);

 // 9: service and resource costs.
 s=base(p,'服务与资源代价：减少阻塞、能耗与切换',9,provenance+'图源：paper/figure2_bar_metrics.pdf。相对PDQN，跨规模均值的累计阻塞用户时间282.664 vs 306.752 user-s/episode，下降7.8526%；总终端能耗6.272299 vs 6.479177 kJ/episode，下降3.19296%；切换频率0.607282 vs 0.712977 handovers/user-minute，下降14.8245%。这些结论只与PDQN比较，不能说所有代价在全部基线中最低。Full-Local只在阻塞子图省略，其切换频率为0。累计阻塞用户时间受用户数影响，不能将跨规模增量直接看成阻塞率增长。');
 await pic(s,'service_metrics',42,120,1196,454);
 txt(s,'累计阻塞用户时间',56,584,365,35,25,C.muted);
 txt(s,'↓ 7.85%',56,623,365,50,39,C.teal,true);
 txt(s,'总终端能耗',457,584,365,35,25,C.muted);
 txt(s,'↓ 3.19%',457,623,365,50,39,C.teal,true);
 txt(s,'每用户每分钟切换',858,584,365,35,25,C.muted);
 txt(s,'↓ 14.82%',858,623,365,50,39,C.teal,true);
 txt(s,'相对 PDQN；先取五规模均值。Full-Local 仅在阻塞子图省略',56,686,1080,22,16,C.muted);

 // 10: bounded conclusion and concrete next evidence.
 s=base(p,'总结与下一步',10,'总结：HAN+PDQN把已有和潜在资源竞争的关系表示引入离散卫星选择与连续部分卸载，当前5训练种子、20–40用户实验支持相对PDQN在任务违约率、成功任务能耗和时延上的改进。提升针对当前完整方法与评估配置，未证明理论最优、HAN独立因果贡献或未见规模泛化。下一阶段优先使用互不重合的验证与最终测试场景，补关系结构和固定动作的受控消融，然后进行任务负载/带宽/MEC算力压力、整网决策耗时及跨规模迁移。当前已有超参数扫描不是资源压力实验；分别按规模训练不是零样本泛化。来源：paper/OUTLINE.md、docs/PAPER_REVIEW_20261029.md、当前多种子统计。');
 txt(s,'方法贡献',56,144,530,45,32,C.blue,true);
 txt(s,'已有 / 潜在竞争的异质关系表示\n候选卫星与连续卸载比例联合决策\n面向长期任务服务质量优化',56,217,650,170,28);
 txt(s,'已验证结果',776,144,448,45,32,C.blue,true);
 txt(s,'20–40 用户 · 5 训练种子',776,216,448,43,27);
 txt(s,'违约率降低',776,294,448,37,26,C.muted);
 txt(s,'21.26%',776,344,448,86,65,C.teal,true);
 txt(s,'相对 PDQN 的跨规模均值',776,439,448,33,22,C.muted);
 txt(s,'下一步验证',56,492,650,45,30,C.blue,true);
 txt(s,'独立测试场景  ·  关系与联合决策消融\n资源压力  ·  决策耗时  ·  未见规模迁移',56,550,1168,94,27);
 return {slides:p.slides.items.length};
}
