// Create a new PlayCanvas script
var InteractionRaycast = pc.createScript('interactionRaycast');

// Define script attributes
InteractionRaycast.attributes.add('cameraEntity', {
    type: 'entity',
    title: 'Camera',
    description: 'The camera entity used for raycasting.'
});

// -- Script Methods --

// initialize() is called once per entity, after all components are created
InteractionRaycast.prototype.initialize = function() {
    // We need a camera entity to be set to work
    if (!this.cameraEntity || !this.cameraEntity.camera) {
        console.error('InteractionRaycast script requires a valid camera entity to be assigned.');
        return;
    }

    // --- Internal State ---
    // These variables act like private members for our script instance
    this._highlightedEntity = null;
    this._originalEmissive = new pc.Color();
    this._highlightColor = new pc.Color(1, 1, 1); // Light color

    // --- Event Listeners ---
    this.app.mouse.on(pc.EVENT_MOUSEDOWN, this.doRaycast, this);

    // Add touch support if available, TODO: debug on mobile
    if (this.app.touch) {
        this.app.touch.on(pc.EVENT_TOUCHSTART, this.doRaycast, this);
    }

    // Clean up listeners when the script is destroyed
    /* this.on('destroy', function() {
        this.app.mouse.off(pc.EVENT_MOUSEDOWN, this.doRaycast, this);
        if (this.app.touch) {
            this.app.touch.off(pc.EVENT_TOUCHSTART, this.doRaycast, this);
        }
    }); */
};

// doRaycast() performs the raycast from camera to screen position
InteractionRaycast.prototype.doRaycast = function(event) {
    let x, y;

    // Check if it's a touch event or a mouse event
    if (event.touches && event.touches.length > 0) {
        x = event.touches[0].x;
        y = event.touches[0].y;
    } else {
        x = event.x;
        y = event.y;
    }

    const cameraComponent = this.cameraEntity.camera;
    const from = cameraComponent.screenToWorld(x, y, cameraComponent.nearClip);
    const to = cameraComponent.screenToWorld(x, y, cameraComponent.farClip);

    // Perform the raycast
    const result = this.app.systems.rigidbody.raycastFirst(from, to);

    // Debug
    console.log("Raycast hit:", result);

    // Always clear the previous highlight
    this._clearHighlight();

    // If we hit something, apply a new highlight
    if (result && result.entity) {
        // We only want to highlight entities that have a render component
        if (result.entity.render) {
            this._applyHighlight(result.entity);
        }
    }
};

// _applyHighlight() changes the material to show a highlight effect
InteractionRaycast.prototype._applyHighlight = function(entity) {
    this._highlightedEntity = entity;
    
    // Check if the entity has mesh instances to highlight
    if (!entity.render.meshInstances || entity.render.meshInstances.length === 0) {
        return;
    }

    const meshInstance = entity.render.meshInstances[0];
    const material = meshInstance.material;

    // Ensure it's a standard material with an emissive property
    if (material && material.emissive) {
        // Save the original color so we can restore it later
        this._originalEmissive.copy(material.emissive);

        // Apply the highlight color
        material.emissive = this._highlightColor;
        material.update();

        // Set a timer to automatically remove the highlight
        setTimeout(() => this._clearHighlight(), 300);
    }
};

// _clearHighlight() restores the original material properties
InteractionRaycast.prototype._clearHighlight = function() {
    if (this._highlightedEntity) {
        if (this._highlightedEntity.render && this._highlightedEntity.render.meshInstances[0]) {
            const material = this._highlightedEntity.render.meshInstances[0].material;
            if (material && material.emissive) {
                material.emissive = this._originalEmissive;
                material.update();
            }
        }
        // Clear the reference
        this._highlightedEntity = null;
    }
};