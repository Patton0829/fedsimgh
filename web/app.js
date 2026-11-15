"use strict";
const $ = id => document.getElementById(id);
const names = ["正常", "内圈故障", "外圈故障", "滚动体故障"];
const statuses = {idle:"等待开始", queued:"正在排队", running:"正在训练", predicting:"逐条诊断", completed:"诊断完成", failed:"任务失败", cancelled:"任务已停止"};
let info, runId=null, offset=0, predictions=[], selected=0, page=0, busy=false, polling=false, filter="all";
let lastStatus="idle", previousRows="", logCount=0;

async function api(path, options={}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}
function notice(message="") { $("notice").textContent=message; $("notice").classList.toggle("hidden",!message); }
function percent(value,digits=1) {return (value*100).toFixed(digits)+"%";}
function element(tag, cls, text) {const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;}
function download(id, name, enabled) {const link=$(id);link.classList.toggle("disabled",!enabled);link.setAttribute("aria-disabled",String(!enabled));if(enabled)link.href=`/api/runs/${runId}/${name}`;else link.removeAttribute("href");}

function renderNodes(state={}) {
  $("nodes").replaceChildren();
  for(let i=0;i<3;i++) {
    const node=(state.nodes||[]).find(n=>n.rank===i);
    const weight=state.metrics?.at(-1)?.weights?.[i];
    const card=element("div","node"+(node?" ready":""));
    const title=element("div","node-head");title.append(element("span",null,`NODE / 0${i}`),element("span"));
    card.append(title,element("span","node-host",node?.hostname || "等待节点分配"),element("span","node-role",i===0?"客户端 + 协调端":"训练客户端"),element("span","node-weight",weight!==undefined?weight.toFixed(3):"—"));
    $("nodes").append(card);
  }
}

function svgElement(name, attributes) {const el=document.createElementNS("http://www.w3.org/2000/svg",name);for(const [k,v] of Object.entries(attributes))el.setAttribute(k,v);return el;}
function renderChart(metrics, rounds) {
  if(!metrics?.length) {$("chart").innerHTML='<div class="chart-empty">提交任务后，训练趋势将在这里更新</div>';return;}
  const svg=svgElement("svg",{viewBox:"0 0 600 135",preserveAspectRatio:"none",role:"img","aria-label":"各轮全局模型测试集准确率"});
  for(let i=0;i<=4;i++) {
    const y=8+i*26;
    svg.append(svgElement("line",{x1:34,y1:y,x2:592,y2:y,stroke:"#edf1f7","stroke-dasharray":"3 4"}));
    const label=svgElement("text",{x:0,y:y+3,fill:"#a2afc3","font-size":9});label.textContent=`${100-i*25}%`;svg.append(label);
  }
  const points=metrics.map(m=>[34+(m.round-1)/Math.max(rounds-1,1)*558,112-m.accuracy*104]);
  const joined=points.map(p=>p.join(",")).join(" ");
  svg.append(svgElement("polygon",{points:`${points[0][0]},112 ${joined} ${points.at(-1)[0]},112`,fill:"#eff3ff"}));
  svg.append(svgElement("polyline",{points:joined,fill:"none",stroke:"#4c72ed","stroke-width":2,"stroke-linejoin":"round"}));
  for(const point of points) svg.append(svgElement("circle",{cx:point[0],cy:point[1],r:2,fill:"#4c72ed"}));
  for(const [x,text] of [[34,"第 1 轮"],[545,`第 ${rounds} 轮`]]) {const label=svgElement("text",{x,y:132,fill:"#a2afc3","font-size":9});label.textContent=text;svg.append(label);}
  $("chart").replaceChildren(svg);
}

function filteredSamples() {
  return (info?.samples||[]).filter(sample=>filter==="all" || (filter==="wrong" ? predictions.find(p=>p.id===sample.id)?.correct===false : sample.true_label===Number(filter)));
}
function renderRows(force=false) {
  const signature=JSON.stringify([predictions,filter,page,selected]);
  if(!force && signature===previousRows)return;
  previousRows=signature;
  const samples=filteredSamples();const pages=Math.max(1,Math.ceil(samples.length/10));page=Math.min(page,pages-1);
  $("prediction-body").replaceChildren();
  for(const sample of samples.slice(page*10,page*10+10)) {
    const prediction=predictions.find(p=>p.id===sample.id);
    const tr=element("tr",sample.id===selected?"selected-row":"");tr.tabIndex=0;
    tr.setAttribute("aria-label",`样本 ${sample.id}，真实类别 ${sample.true_name}`);
    const choose=()=>{selected=sample.id;renderRows();renderSignal();};tr.onclick=choose;tr.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();choose();}};
    tr.append(element("td",null,`SIG-${String(sample.id).padStart(3,"0")}`),element("td",null,sample.true_name));
    const pred=element("td");pred.append(element("span",prediction?"pred":"pending",prediction?.predicted_name||"等待诊断"));
    const result=element("td");result.append(element("span",prediction?(prediction.correct?"correct":"wrong"):"pending",prediction?(prediction.correct?"✓ 正确":"× 错误"):"—"));
    tr.append(pred,element("td",null,prediction?percent(prediction.confidence):"—"),result);$("prediction-body").append(tr);
  }
  if(!samples.length) {const tr=element("tr");const td=element("td","pending","暂无匹配的诊断结果");td.colSpan=5;tr.append(td);$("prediction-body").append(tr);}
  $("page-summary").textContent=`共 ${samples.length} 条${filter==="all"?"测试":"匹配"}样本`;
  $("page-number").textContent=`${page+1} / ${pages}`;
  $("prev-page").disabled=page===0;$("next-page").disabled=page>=pages-1;
  $("prediction-count").textContent=`${predictions.length} / ${info?.samples?.length||80}`;
}

function renderSignal() {
  const sample=info?.samples?.find(s=>s.id===selected);if(!sample)return;
  const prediction=predictions.find(p=>p.id===selected);
  $("selected-sample").textContent=`SIGNAL / ${String(selected).padStart(3,"0")}`;
  const wave=sample.waveform;const max=Math.max(...wave.map(Math.abs),1);
  const points=wave.map((v,i)=>`${i/(wave.length-1)*360},${50-v/max*42}`).join(" ");
  $("waveform").replaceChildren(svgElement("polyline",{points,fill:"none",stroke:"#6486d9","stroke-width":1.4}));
  $("selected-result").textContent=prediction?.predicted_name||"等待模型预测";
  $("probabilities").replaceChildren();
  names.forEach((name,i)=>{const value=prediction?.probabilities[i];const row=element("div","prob-row"+(prediction?.predicted_label===i?" winner":""));const track=element("span","prob-track");const fill=element("i");fill.style.width=`${(value||0)*100}%`;track.append(fill);row.append(element("span",null,name),track,element("span",null,value===undefined?"—":percent(value,0)));$("probabilities").append(row);});
}

function appendLogs(events) {
  if(!events.length)return;
  if(!logCount)$("log-output").replaceChildren();
  const kinds={start:"START",client:"CLIENT",aggregate:"AGG",metric:"METRIC",predict_start:"INFER",prediction:"PREDICT",complete:"DONE",error:"ERROR"};
  const fragment=document.createDocumentFragment();
  for(const event of events) {
    const line=element("div",`log-line ${event.kind}`);
    const time=new Date(event.time*1000).toLocaleTimeString("zh-CN",{hour12:false});
    line.append(element("span","log-time",time),element("span","log-kind",kinds[event.kind]||"INFO"),element("span","log-message",event.message));fragment.append(line);logCount++;
  }
  $("log-output").append(fragment);
  if($("auto-scroll").checked)$("log-output").scrollTop=$("log-output").scrollHeight;
}

function renderState(data) {
  const state=data.state;lastStatus=state.status;
  const active=["queued","running","predicting"].includes(state.status);
  $("run-badge").textContent=statuses[state.status]||state.status;$("run-badge").className=`status-badge ${state.status}`;
  $("start-button").disabled=active||busy;$("stop-button").disabled=!active||busy;
  for(const input of $("train-form").querySelectorAll("input"))input.disabled=active;
  const rounds=state.rounds||Number($("rounds").value);
  $("round-value").replaceChildren(document.createTextNode(String(state.round||0).padStart(2,"0")),element("small",null,`/ ${rounds} 轮`));
  $("elapsed-value").textContent=state.elapsed?`CPU 作业耗时 ${state.elapsed.toFixed(1)} 秒`:active?"等待计算节点启动":"CPU 执行 · 尚未开始";
  const metric=state.final_metrics||state.metrics?.at(-1);
  $("accuracy-value").replaceChildren(document.createTextNode(metric?(metric.accuracy*100).toFixed(1):"—"),element("small",null,"%"));
  $("f1-value").textContent=metric?`Macro-F1 ${metric.macro_f1.toFixed(3)} · 80 条测试信号`:"等待模型评估";
  const progress=(state.round||0)/rounds*100;
  $("progress-bar").style.width=`${progress}%`;$("progress-percent").textContent=`${progress.toFixed(0)}%`;
  $("progress-label").textContent=state.status==="completed"?"训练与逐条诊断已完成":state.status==="running"?`联邦训练中 · 第 ${state.round} / ${rounds} 轮`:statuses[state.status];
  $("job-label").textContent=data.submission?.job_id?`JOB ${data.submission.job_id}`:data.run_id?"LOCAL · 3 PROCESSES":"JOB —";
  $("terminal-title").textContent=data.run_id?`faultdiag / ${data.run_id}`:"faultdiag / awaiting-job";
  const scheduler=data.submission?.job_id?`Slurm ${state.scheduler_state||"SUBMITTED"} · ${state.scheduler_message||"等待资源分配"}`:info?.mode==="local"?"本地验证：三个独立 CPU 进程模拟三个节点。":"Slurm：3 节点 × 1 任务，无需 GPU。";
  $("scheduler-message").textContent=state.nodes?.length===3?state.nodes.map(n=>`节点${n.rank} ${n.hostname}`).join(" · "):scheduler;
  if(state.message)notice(state.message);
  renderNodes(state);renderChart(state.metrics,rounds);
  predictions=data.predictions||[];renderRows();renderSignal();appendLogs(data.events||[]);
  download("csv-download","predictions.csv",predictions.length>0);download("logs-download","logs.txt",!!data.run_id);
}

async function poll() {
  if(polling)return;polling=true;
  try {
    const query=new URLSearchParams({offset:String(offset)});if(runId)query.set("run",runId);
    const data=await api(`/api/state?${query}`);
    if(data.run_id!==runId){runId=data.run_id;offset=0;logCount=0;}
    offset=data.offset;renderState(data);
  } catch(error){notice(`连接失败：${error.message}。正在自动重试。`);}
  finally{polling=false;}
}

$("train-form").onsubmit=async event=>{
  event.preventDefault();if(busy)return;busy=true;$("start-button").disabled=true;notice();
  try {
    const options={rounds:Number($("rounds").value),epochs:Number($("epochs").value),lr:Number($("lr").value),tau:Number($("tau").value),history_lambda:Number($("history-lambda").value)};
    const data=await api("/api/runs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(options)});
    runId=data.run_id;offset=0;predictions=[];page=0;logCount=0;previousRows="";$("log-output").replaceChildren(element("div","log-placeholder",info.mode==="slurm"?"Slurm 作业已提交。等待资源分配后，训练日志将在这里更新。":"三个 CPU 客户端正在启动…"));
    await poll();
  }catch(error){notice(error.message);}
  finally{busy=false;$("start-button").disabled=["queued","running","predicting"].includes(lastStatus);}
};
$("stop-button").onclick=async()=>{if(!runId||busy)return;busy=true;$("stop-button").disabled=true;try{await api(`/api/runs/${runId}/stop`,{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});await poll();}catch(error){notice(error.message);}finally{busy=false;$("stop-button").disabled=!["queued","running","predicting"].includes(lastStatus);}};
$("history-lambda").oninput=()=>$("lambda-value").textContent=Number($("history-lambda").value).toFixed(2);
$("result-filter").onchange=event=>{filter=event.target.value;page=0;renderRows();};
$("prev-page").onclick=()=>{page--;renderRows();};$("next-page").onclick=()=>{page++;renderRows();};
document.querySelectorAll(".nav-item").forEach(link=>link.addEventListener("click",()=>{document.querySelectorAll(".nav-item").forEach(l=>l.classList.remove("selected"));link.classList.add("selected");}));
(async()=>{try{info=await api("/api/info");$("mode-label").textContent=info.mode==="slurm"?"Slurm · 三节点 CPU":"本地验证 · 三进程 CPU";if(info.mode==="slurm"&&!info.slurm_available)notice("当前主机没有 sbatch。请在 Slurm 登录节点启动，或使用 --mode local 验证。");renderNodes();renderRows(true);renderSignal();await poll();}catch(error){notice(error.message);}setInterval(poll,1000);})();
