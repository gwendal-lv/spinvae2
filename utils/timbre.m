function rc = timbre(directories_list_file)
    disp(strcat('Script starts: ', datestr(now, 'yy/mm/dd-HH:MM:SS')));

    disp('Input args file: ')
    disp(directories_list_file)

    % FIXME replace audio_root_path
    %sub_folders = readlines(directories_list_file)  % Matlab 2020...
    fid = fopen(directories_list_file);
    sub_folders = {};  % Empty array
    tline = fgetl(fid);
    while ischar(tline)
        sub_folders(end+1) = {tline};
        tline = fgetl(fid);
    end
    fclose(fid);


    % Process folders one by one
    for folder_index = 1 : length(sub_folders)

        soundsDirectory = sub_folders{folder_index};  % Get the contents of a cell

        % Parts of: https://github.com/VincentPerreault0/timbretoolbox/blob/master/doc/Full_Config_Example.m
        singleFileName = '';
        csvDirectory = soundsDirectory;
        matDirectory = '';  % soundsDirectory;
        pltDirectory = '';

        sndConfig = struct();


        evalConfig = struct();

        % OK if not audio signal descriptor?
        evalConfig.AudioSignal.NoDescr = struct();

        %
        evalConfig.TEE = struct();
        evalConfig.TEE.CutoffFreq = 5;

        evalConfig.TEE.Att = struct();          % Specified to be evaluated/plotted
        evalConfig.TEE.Att.Method = 3;          % params shared with Dec, Rel, LAT, AttSlope, DecSlope
        evalConfig.TEE.Att.NoiseThresh = 0.15;  %                   (LAT = Log-Attack Time)
        evalConfig.TEE.Att.DecrThresh = 0.4;
        evalConfig.TEE.TempCent = struct();     % Specified to be evaluated/plotted
        evalConfig.TEE.TempCent.Threshold = 0.15;
        evalConfig.TEE.EffDur = struct();       % Specified to be evaluated/plotted
        evalConfig.TEE.EffDur.Threshold = 0.4;
        evalConfig.TEE.FreqMod = struct();      % Specified to be evaluated/plotted
        evalConfig.TEE.FreqMod.Method = 'fft';  % shared with TEE.AmpMod; require Dec and Rel

        evalConfig.TEE.RMSEnv = struct();       % Specified to be evaluated/plotted
        evalConfig.TEE.RMSEnv.HopSize_sec = 0.0029;
        evalConfig.TEE.RMSEnv.WinSize_sec = 0.0232;

        evalConfig.STFT = struct();             % Specified to be evaluated/plotted
        evalConfig.STFT.DistrType = 'pow';
        evalConfig.STFT.HopSize_sec = 0.0058;
        evalConfig.STFT.WinSize_sec = 0.0232;
        evalConfig.STFT.WinType = 'hamming';
        evalConfig.STFT.FFTSize = 1024;
        % If no descriptors are specified in the evalConfig.STFT structure, all descriptors will be evaluated

        evalConfig.ERB = struct();              % Specified to be evaluated/plotted
        evalConfig.ERB.HopSize_sec = 0.0058;
        evalConfig.ERB.Method = 'fft';
        evalConfig.ERB.Exponent = 1/4;
        % If no descriptors are specified in the evalConfig.ERB structure, all descriptors will be evaluated

        evalConfig.Harmonic = struct();             % Specified to be evaluated/plotted
        evalConfig.Harmonic.Threshold = 0.3;
        evalConfig.Harmonic.NHarms = 20;
        evalConfig.Harmonic.HopSize_sec = 0.025;
        evalConfig.Harmonic.WinSize_sec = 0.1;
        evalConfig.Harmonic.WinType = 'blackman';
        evalConfig.Harmonic.FFTSize = 32768;
        % If no descriptors are specified in the evalConfig.Harmonic structure, all descriptors will be evaluated

        csvConfig = struct();
        csvConfig.Directory = csvDirectory;
        csvConfig.TimeRes = 10;
        % Default grouping: 'sound' (1 CSV file / audio file)
        csvConfig.Grouping = 'sound';               % group by descriptor: replace with 'descr'
        % default: {'stats', 'ts'}
        csvConfig.ValueTypes = {'stats'};     % only statistics: replace with 'stats'
        %%%                                 % only time series: replace with 'ts'
        matConfig = struct();
        matConfig.Directory = matDirectory;

        plotConfig = struct();
        plotConfig.Directory = pltDirectory;
        plotConfig.TimeRes = 0;

        if ~isdir(soundsDirectory)
            error('soundsDirectory must be a valid directory.');
        end
        if ~isempty(singleFileName)
            filelist.name = singleFileName;
        else
            filelist = dir(soundsDirectory);
        end
        acceptedFormats = {'wav', 'ogg', 'flac', 'au', 'aiff', 'aif', 'aifc', 'mp3', 'm4a', 'mp4'};
        for i = 1:length(filelist)
            [~, fileName, fileExt] = fileparts(filelist(i).name);
            if ~isempty(fileName) && fileName(1) ~= '.' && (any(strcmp(fileExt(2:end), acceptedFormats)) || (length(filelist) == 1 && strcmp(fileExt(2:end), 'raw')))
                sound = SoundFile([soundsDirectory '/' fileName fileExt], sndConfig);
                % catch sound.Eval Error
                sound_was_eval = false;
                try
                    sound.Eval(evalConfig);
                    sound_was_eval = true;
                % when timbre toolbox bugs with low-volume samples: general Error, nothing more specific....
                % so we'll have to catch all Error
                catch Error
                    stats_file = strcat(soundsDirectory, '/', fileName, '_stats.csv')   % FIXME temp
                    fid = fopen(stats_file, 'w');
                    warning("000000000000000000000 - sound.Eval has raised an Error - an empty .csv file will be written - 00000000000000000000000000000");
                    fprintf(fid, "Evaluation Error");
                    fclose(fid);
                end

                if sound_was_eval
                    % write files - code from the original timbre toolbox demo file
                    if ~isempty(csvDirectory)
                        sound.ExportCSV(csvConfig);
                    end
                    if ~isempty(matDirectory)
                        sound.Save(matConfig);
                    end
                    if ~isempty(pltDirectory)
                        sound.Plot(plotConfig);
                        close all;
                        clc
                    end
                end
                clear 'sound';
            end

            % to try to solve Exception in thread "AWT-EventQueue-0" java.lang.OutOfMemoryError: Java heap space
            % (seems related to FigureComponent... calls, even when we don't ask for plots)
            %      ---> does not solve the issue
            close all;
            clc;
        end


    end % subfolder-by-subfolder processing


    fprintf('Processed %d audio folders.\n', length(sub_folders))
    disp(strcat('Script ends: ', datestr(now, 'yy/mm/dd-HH:MM:SS')));
    disp('timbre.m   EXIT_SUCCESS');
    rc = 0;
end
