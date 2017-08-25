function sPrm = ParameterSetup(hParent,tree)
% Tree GUI for parameter configuration
% 
% sPrm = ParameterSetup(hParent,tree)
%
% hParent: not a real parent, just figure over which window is centered
% tree: TreeNode tree, generated by parseConfigYaml()
%
% sPrm: If "Apply" is pushed, parameter structure; otherwise, []

assert(isscalar(hParent) && ishandle(hParent));

hFig = figure('Visible','off','ToolBar','none','menubar','none',...
  'Name','CPR Tracking Parameters');
hFig.Position(3) = 400;
centerOnParentFigure(hFig,hParent);

assert(isa(tree,'TreeNode') && isscalar(tree));
% for cosmetic purposes, don't include Dummy/Root node in propertiesGUI
assert(strcmp(tree.Data.Field,'ROOT'));
rootnode = tree;
children = tree.Children;
rootnode.Children = [];
propertiesGUI2(hFig,children);
setappdata(hFig,'rootnode',rootnode); % save to glue rootnode back on at output

h = findall(hFig,'type','hgjavacomponent');
LOFF = 0.025;
BOFF = 0.1;
pos = h.Position;
set(h,'Units','normalized','Position',[LOFF pos(2)+BOFF 1-2*LOFF pos(4)-BOFF]);
BOFF2 = 0.01;
BTNWIDTH = .2;
BTNGAP = .01;
hApply =  uicontrol('String','Apply','Units','normalized',...
  'Pos',[0.5-BTNWIDTH-BTNGAP/2 BOFF2 BTNWIDTH BOFF-2*BOFF2],...
  'FontUnits','pixels','fontsize',16,...
  'Tag','pbApply','Callback',@(s,e)cbkApply(s,e,hFig));
hCncl =  uicontrol('String','Cancel','Units','normalized',...
  'Pos',[.5+BTNGAP/2 BOFF2 BTNWIDTH BOFF-2*BOFF2],...
  'FontUnits','pixels','fontsize',16,...
  'Tag','pbCncl','Callback',@(s,e)cbkCncl(s,e,hFig));

hFig.Visible = 'on';
  
hObj = HandleObj;
setappdata(hFig,'outputObj',hObj);
uiwait(hFig);
sPrm = hObj.data;

function cbkApply(~,~,hFig)
t = getappdata(hFig,'mirror');
rootnode = getappdata(hFig,'rootnode');
rootnode.Children = t;
s = rootnode.structize();
hObj = getappdata(hFig,'outputObj');
hObj.data = s;
delete(hFig);

function cbkCncl(~,~,hFig)
delete(hFig);
