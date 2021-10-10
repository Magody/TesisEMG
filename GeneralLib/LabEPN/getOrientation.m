function orientation = getOrientation(userData, user_folder)
    syncro        = 4;
    energy_umbral = 0.2;
    if syncro>0
        for x=1:150
            gesto_=userData.training{x}.gestureName;
            if gesto_=="waveOut"
                location_=x;
                break;
            end
        end
        elec_=zeros(1,syncro);
        aux=1;
        energy_order=zeros(syncro,8);
        % =======================================================
        %                     WITH ROTATION
        % =======================================================
        simulate_Rotation = simulateRotation();

        for goto_=location_:location_+syncro-1

            emgData             = userData.training{goto_}.emg(:,simulate_Rotation);
            Index_              = userData.training{goto_}.groundTruthIndex;
            Index_high_         = Index_(1,2);
            emgData             = emgData(Index_high_ - 255:Index_high_,:);
            energy_wm           = WMoos_F5(emgData');
            energy_order(aux,:) = energy_wm;
            [~,max_energy]      = max((energy_wm));
            elec_(1,aux)        = max_energy;
            aux = aux+1;
        end
        ref_partial         = histcounts(elec_(1,:),1:(8+1));
        [~,ref]             = max(ref_partial);
        xyz                 = ref;
    else
        xyz_aux             = simulateRotation();
        xyz                 = xyz_aux(:,1);
    end

    % ================== Umbral =========================

    calibration_umbral=zeros(8,syncro);
    for o=1:syncro
        waveout_pure=userData.sync{o,1}.emg(:,simulate_Rotation);
        umbral_envelope_wm=WMoos_F5(waveout_pure');
        calibration_umbral(:,o)=umbral_envelope_wm;
    end
    sequence_=WM_X(xyz);
    calibration_umbral=calibration_umbral';
    calibration_umbral=calibration_umbral(:,sequence_);
    mean_umbral=calibration_umbral;
    mean_umbral=mean(mean_umbral,1);
    val_umbral_high = energy_umbral*sum(mean_umbral(1:4))/4;
    val_umbral_low  = energy_umbral*sum(mean_umbral(5:8))/4;

    orientation = {user_folder, xyz, val_umbral_low, val_umbral_high, simulate_Rotation(:,1)};
end

