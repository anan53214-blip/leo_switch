import fs from 'node:fs/promises';
import {constants} from 'node:fs';
import {Presentation,PresentationFile} from 'file:///C:/Users/19704/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const root='D:/program/LEO_switch';
const build=root+'/tmp/pptx_report_20261030';
const skill='C:/Users/19704/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
process.env.RUNTIME_NODE_MODULES='C:/Users/19704/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {finalizePresentation}=await import('file:///'+skill+'/container_tools/artifact_tool_utils.mjs');
const proto=JSON.parse(await fs.readFile(build+'/deck.json','utf8'));
for(const im of proto.images??[])if(im.data)im.data=Uint8Array.from(Object.values(im.data));
const presentation=await Presentation.load(proto);
await (await PresentationFile.exportPptx(presentation)).save(build+'/candidate.pptx');
const out=root+'/output/presentations/HAN_PDQN_项目汇报_20261030.pptx';
await fs.mkdir(root+'/output/presentations',{recursive:true});
// Windows host disallows hard-link publication. Retain all validations and
// exclusive destination creation, copying the already-validated bytes instead.
const originalLink=fs.link.bind(fs);
fs.link=async (source,destination)=>{
 try{return await originalLink(source,destination);}catch(error){
  if(error.code!=='EPERM'||destination.replaceAll('\\','/')!==out)throw error;
  return fs.copyFile(source,destination,constants.COPYFILE_EXCL);
 }
};
const result=await finalizePresentation({
 workspaceDir:root,candidatePath:build+'/candidate.pptx',finalPath:out,
 pythonExecutable:'C:/Users/19704/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','2','--require-native-table-slide','4','--require-native-table-slide','6'],
 explicitTotalSlideCount:10,requiredNativeTableOwnerSlides:[2,4,6],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:['Microsoft YaHei']},
 verifyArtifactToolImport:true,receiptPath:build+'/validation.json'
});
console.log(JSON.stringify({finalPath:result.finalPath,byteCount:result.byteCount,sha256:result.finalSha256,validation:result.receiptPath}));
