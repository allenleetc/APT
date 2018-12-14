classdef BgTrainWorkerObjLocalFilesys < BgTrainWorkerObj
  %
  % 
  % 1. Training artifacts written to local filesys
  % 2. Training killed by sending message, polling to confirm, and touching
  % filesystem tok
  %
  
  properties
    jobID % [nview] bsub jobID; or docker cellstr containerID
    
    killPollIterWaitTime = 1; % sec
    killPollMaxWaitTime = 12; % sec
  end
  
  methods (Abstract)
    killJob(obj,jID) % kill a single job. jID is scalar jobID
    fcn = makeJobKilledPollFcn(obj,jID) % create function that returns true when job is confirmed killed. jID is scalar jobID
    createKillToken(obj,killtoken) % create/touch filesystem KILL token. killtoken is full linux path
  end
  
  methods
    
    function obj = BgTrainWorkerObjLocalFilesys(nviews,dmcs)
      obj@BgTrainWorkerObj(nviews,dmcs);      
    end
    
    function tf = fileExists(~,file)
      tf = exist(file,'file')>0;
    end
    
    function tf = errFileExistsNonZeroSize(~,errFile)
      tf = BgTrainWorkerObjLocalFilesys.errFileExistsNonZeroSizeStc(errFile);
    end
        
    function s = fileContents(~,file)
      if exist(file,'file')==0
        s = '<file does not exist>';
      else
        lines = readtxtfile(file);
        s = sprintf('%s\n',lines{:});
      end
    end
    
    function killProcess(obj)
      dmcs = obj.dmcs;
      killfiles = {dmcs.killTokenLnx};
      jobids = obj.jobID;
      nvw = obj.nviews;
      assert(isequal(nvw,numel(jobids),numel(killfiles)));
      
      for ivw=1:nvw
        obj.killJob(jobids(ivw));
      end

      iterWaitTime = obj.killPollIterWaitTime;
      maxWaitTime = obj.killPollMaxWaitTime;

      for ivw=1:nvw
        fcn = obj.makeJobKilledPollFcn(jobids(ivw));
        tfsucc = waitforPoll(fcn,iterWaitTime,maxWaitTime);
        
        if ~tfsucc
          warningNoTrace('Could not confirm that training job was killed.');
        else
          % touch KILLED tokens i) to record kill and ii) for bgTrkMonitor to
          % pick up
          
          kfile = killfiles{ivw};
          obj.createKillToken(kfile);
        end
        
        % bgTrnMonitor should pick up KILL tokens and stop bg trn monitoring
      end
    end
        
  end
    
  methods (Static)
    function tfErrFileErr = errFileExistsNonZeroSizeStc(errFile)
      tfErrFileErr = exist(errFile,'file')>0;
      if tfErrFileErr
        direrrfile = dir(errFile);
        tfErrFileErr = direrrfile.bytes>0;
      end
    end
  end
  
end