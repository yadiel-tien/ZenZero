// 在 node 里跑 admin.html 的页面 JS（DOM/canvas/fetch 全 stub，fetch 走真实服务）
const fs=require('fs');
let js=fs.readFileSync('.audit/app.js','utf8');
// 剥离会立刻发起的 boot 依赖前，先搭 DOM stub
const els={};
function mkEl(id){ const e = {id, innerHTML:'', textContent:'', value:'', checked:false,
  get outerHTML(){ return `<option value="${this.value}" title="${this.title||''}">${this.textContent}</option>`; },
  disabled:false, selectedIndex:0, style:{}, classList:{toggle(){},add(){},remove(){},contains:()=>false}, dataset:{},
  options:[], appendChild(){}, setAttribute(k,v){this[k]=v;}, title:'', addEventListener(){}, onclick:null, onchange:null,
  getContext:()=>new Proxy({},{get:()=>()=>({})}), clientWidth:800, clientHeight:200, scrollTop:0, scrollHeight:0 };
  return els[id] ||= e; }
global.document={ getElementById:id=>mkEl(id), createElement:t=>mkEl('tmp_'+t+Math.random()),
  querySelectorAll:()=>[], addEventListener:()=>{}, body:mkEl('body'), documentElement:{setAttribute(){},getAttribute:()=>null,dataset:{}},
  createElementNS:()=>mkEl('svgtmp'), fonts:{ready:Promise.resolve()} };
global.window=global; global.navigator={}; global.location={hash:'#artifacts',search:''};
global.localStorage={getItem:()=>null,setItem(){}}; global.matchMedia=()=>({matches:false,addEventListener(){}});
global.requestAnimationFrame=f=>f(); global.getComputedStyle=()=>({getPropertyValue:()=>''});
global.fetch=async(u,o)=>{ const r=await import('node:fetch').then(()=>fetch); return _f(u,o); };
const http=require('node:http');
global._realfetch=(u,o={})=>new Promise((res,rej)=>{ const url=new URL(u,'http://127.0.0.1:5000');
  const rq=http.request(url,{method:o.method||'GET',headers:o.headers||{}},rp=>{let b='';rp.on('data',c=>b+=c);rp.on('end',()=>res({ok:rp.statusCode<400,status:rp.statusCode,text:async()=>b,json:async()=>JSON.parse(b)}));});
  rq.on('error',rej); if(o.body)rq.write(o.body); rq.end();});
global.fetch=global._realfetch;
global.Chart=undefined;
global.addEventListener=()=>{}; global.removeEventListener=()=>{};
global.matchMedia=global.matchMedia||(()=>({matches:false,addEventListener(){},removeEventListener(){}}));
global.EventSource=class{addEventListener(){}close(){}};
global.ResizeObserver=class{observe(){}unobserve(){}disconnect(){}};
try{ (0,eval)(js); }catch(e){ console.log('加载页面JS即抛错:', e.message); process.exit(1); }
(async()=>{
  for(const [name,fn] of [['loadArtifacts',loadArtifacts],['loadPools',()=>loadPools('ChineseChess')],
                          ['loadBuffers',loadBuffers],['loadProfiles',loadProfiles],['tick',tick],['loadLogList',loadLogList]]){
    try{ await fn(); console.log('✓', name); }
    catch(e){ console.log('✗', name, '抛错:', e.message); }
  }
  console.log('--- t_pools 最终内容:', (els['t_pools']||{innerHTML:'(未渲染)'}).innerHTML.replace(/<[^>]+>/g,'·').slice(0,160));
  console.log('--- f_buffer options:', String((els['f_buffer']||{}).innerHTML).replace(/<[^>]+>/g,'|').slice(0,400));
})();
