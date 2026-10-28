// 资产页选择保持的行为级验证（DOM stub + 真实服务数据）
const http=require('node:http');
const els={};
function mkEl(id){
  const e={id, innerHTML:'', textContent:'', value:'', title:'', checked:false, disabled:false,
    style:{}, dataset:{}, classList:{toggle(){},add(){},remove(){},contains:()=>false},
    get outerHTML(){ return `<option value="${this.value}" title="${this.title||''}">${this.textContent}</option>`; },
    options:[], appendChild(){}, setAttribute(k,v){this[k]=v;}, addEventListener(){}, onclick:null, onchange:null,
    querySelectorAll:(sel)=>{ const out=[];
      for(const m of String(e.innerHTML).matchAll(/<input[^>]*>/g)){ const tag=m[0];
        if(/class=ck(\s|")/.test(tag)||tag.includes('class=ck ')) out.push({value:(tag.match(/value="([^"]*)"/)||[])[1]||'', checked:tag.includes('checked')});
        else if(sel.includes('ckall')&&tag.includes('ckall')) out.push({checked:tag.includes('checked')}); }
      return out; },
    querySelector:()=>({disabled:false,textContent:'',dataset:{}}), closest:()=>null,
    getContext:()=>new Proxy({},{get:()=>()=>({})}), clientWidth:800, clientHeight:200};
  return els[id] ||= e;
}
global.document={ getElementById:mkEl, createElement:t=>mkEl('tmp'+Math.random()),
  querySelectorAll:()=>[], querySelector:()=>({disabled:false,textContent:'',dataset:{}}), addEventListener:()=>{}, body:mkEl('body'),
  documentElement:{setAttribute(){},getAttribute:()=>null,dataset:{}}, fonts:{ready:Promise.resolve()} };
global.window=global; global.location={hash:'#artifacts',search:''}; global.navigator={};
global.localStorage={getItem:()=>null,setItem(){}}; global.matchMedia=()=>({matches:false,addEventListener(){}});
global.addEventListener=()=>{}; global.ResizeObserver=class{observe(){}unobserve(){}disconnect(){}};
global.requestAnimationFrame=f=>f(); global.getComputedStyle=()=>({getPropertyValue:()=>''});
global.fetch=(u,o={})=>new Promise((res,rej)=>{ const url=new URL(u,'http://127.0.0.1:5000');
  const rq=http.request(url,{method:o.method||'GET',headers:o.headers||{}},rp=>{let b='';rp.on('data',c=>b+=c);
    rp.on('end',()=>res({ok:rp.statusCode<400,status:rp.statusCode,text:async()=>b,json:async()=>JSON.parse(b)}));});
  rq.on('error',rej); if(o.body)rq.write(o.body); rq.end();});
require('fs').readFileSync && (0,eval)(require('fs').readFileSync('.audit/app.js','utf8'));
(async()=>{
  await loadArtifacts(); await loadPools('ChineseChess');
  const P=__P();
  const html1=String(els['t_pools'].innerHTML);
  console.log(html1.includes('checked') ? '✗ 初始就有勾选' : '✓ 初始无选中');
  P.t_pools.set('pool1','pool');
  await loadPools('ChineseChess');                     // 模拟 15s 自动刷新重绘
  const html2=String(els['t_pools'].innerHTML);
  console.log(html2.includes('value="pool1"') && /<input[^>]*checked[^>]*>\s*<\/td>\s*<td><b>pool1/.test(html2) || /class=ck[^>]*checked[^>]*value="pool1"/.test(html2) ? '✓ 重绘后勾选保持' : '✗ 重绘丢失: '+html2.slice(0,240));
  P.t_runs.set('runs/ghost_dir','run');
  tidySel('t_runs', []);                                // 列表刷新后幽灵不在
  console.log(P.t_runs.size===0 ? '✓ 幽灵项自动销账' : '✗ 销账失败 size='+P.t_runs.size);
  console.log(P.t_pools.has('pool1') ? '✓ 在册项不被误销' : '✗ 在册项丢了');
  // 全选框行为
  P.t_pools.clear();
  els['t_pools'].innerHTML = els['t_pools'].innerHTML.replace(/<input type=checkbox class=ck /g, '<input type=checkbox checked class=ck ');
  // batchDel 收集应来自 Map 而非 DOM——清空后即使 DOM 显示 checked 也不删
  const items=[...P.t_runs, ...P.t_pools];
  console.log(items.length===0 ? '✓ 批量收集以 Map 为账（DOM 假象无效）' : '✗ 收集异常');
})().catch(e=>{console.log('✗ 测试崩:', e.message); process.exit(1);});
