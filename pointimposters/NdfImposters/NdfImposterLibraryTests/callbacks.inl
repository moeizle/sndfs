// when someone changes the value in the tweakbar
void TW_CALL twSetTime(const void *value, void *clientData) {
    unsigned int newTime = *static_cast<const unsigned int*>(value);
    if (newTime != currentTime) {
        currentTime = newTime;
        // TODO: load data and stuff
    }
}

// when the tweakbar is refreshed, it gets the current value using this
void TW_CALL twGetTime(void *value, void *clientData) {
    *static_cast<unsigned int *>(value) = currentTime;
}

// when someone changes the value in the tweakbar
void TW_CALL twSetBinning(const void *value, void *clientData) {
    binning_mode = *static_cast<const int*>(value);

    if (binning_mode == 0) {
        std::cout << "Binning using old method" << std::endl;
    } else if (binning_mode == 1) {
        std::cout << "Binning using Spherical Coordinates binning" << std::endl;
    } else if (binning_mode == 2) {
        std::cout << "Binning using Lambert Azimuthal Equal-Area projection binning" << std::endl;
    }
    //initialize();
    clear_NDF_Cache();
    tile_based_culling(false);
    tile_based_panning(false);
    update_page_texture();
    computeBinAreas();
    preIntegrateBins();
    reset();
    display();
}

// when the tweakbar is refreshed, it gets the current value using this
void TW_CALL twGetBinning(void *value, void *clientData) {
    *static_cast<int *>(value) = binning_mode;
}

// when someone changes the value in the tweakbar
void TW_CALL twSetRadiusScaling(const void *value, void *clientData) {
    float newScaling = *static_cast<const float*>(value);
    if (newScaling != particleScale) {
        particleScale = newScaling;
        //particleScale *= 1.1f;
        clear_NDF_Cache();

        tile_based_culling(false);
        tile_based_panning(false);
        update_page_texture();
        reset();
        display();
    }
}

// when the tweakbar is refreshed, it gets the current value using this
void TW_CALL twGetRadiusScaling(void *value, void *clientData) {
    *static_cast<float *>(value) = particleScale;
}

void TW_CALL twSetQuat(const void *value, void *clientData) {
    glm::quat tmp(static_cast<const float*>(value)[0], static_cast<const float*>(value)[1], static_cast<const float*>(value)[2], static_cast<const float*>(value)[3]);
    glm::quat old(quat[0], quat[1], quat[2], quat[3]);
    if (tmp != old) {
        memcpy(quat, value, 4 * sizeof(float));
        
        rotate_data(0.0f, 0.0f); // values are ignored anyway if ATB does the rotation
        initialize();

        cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
        camPosi = CameraPosition(cameraRotation, cameraDistance);
        camPosi += cameraOffset;
        camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

        tile_based_culling(false);
        //tile_based_panning(false);
        update_page_texture();
        reset();
        display();
    }
}

// when the tweakbar is refreshed, it gets the current value using this
void TW_CALL twGetQuat(void *value, void *clientData) {
    memcpy(value, quat, 4 * sizeof(float));
}