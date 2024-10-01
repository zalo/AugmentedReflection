import * as THREE from '../node_modules/three/build/three.module.js';
import Stats from '../node_modules/three/examples/jsm/libs/stats.module.js';
import { GUI } from '../node_modules/three/examples/jsm/libs/dat.gui.module.js';
import { OrbitControls } from '../node_modules/three/examples/jsm/controls/OrbitControls.js';
import { FaceLandmarker, PoseLandmarker, FilesetResolver, DrawingUtils } from "../node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";
//import { FaceLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

/** The fundamental set up and animation structures for 3D Visualization */
export default class World {

    constructor() {
        this._setupWorld();

        // gui 
        this.gui = new GUI();
        this.params = { FoV: 60, DepthScalar: 0.2 };
        this.gui.add(this.params, 'FoV').min(30).max(100).name('Webcam FoV');
        this.gui.add(this.params, 'DepthScalar').min(0.05).max(0.5).name('DepthScalar');

        // instanced spheres
        this.sphereGeometry = new THREE.SphereGeometry(1.0, 6, 6);
        this.solidMat = new THREE.MeshPhongMaterial({ wireframe: false });
        this.landmarks = new THREE.InstancedMesh(this.sphereGeometry, this.solidMat, 468);
        this.landmarks.receiveShadow = true;
        this.landmarks.castShadow = true;
        this.scene.add(this.landmarks);
        this.offset = new THREE.Vector3();
        this.rawPoints = [];
        this.filteredPoints = [];

        this.hashParams = new URLSearchParams(window.location.hash.substring(1));
        this.useFaceLandmarking = !(this.hashParams.has('bodyLandmarking') &&
                                    this.hashParams.get('bodyLandmarking').toLowerCase() === 'true');

        // Construct the Face Tracking Pipeline
        this.setupFaceTracking();
    }

    async setupFaceTracking() {
        this.faceMesh = await this.createLandmarker();
        this.videoIsPlaying = false;
        /** @type {HTMLMediaElement} */
        this.video = document.getElementById("video");
        navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
            this.video.srcObject = stream;
            this.video.addEventListener("loadeddata", () =>{ 
                this.videoIsPlaying = true; 
            
                // Create a camera for the webcam
                this.lastVideoTime = -1;
                this.webcamera = new THREE.Group();
                this.webcamera.g = new THREE.Vector2(this.video.videoWidth, this.video.videoHeight);
                this.webcamera.camera = new THREE.PerspectiveCamera( 60, this.webcamera.g.width / this.webcamera.g.height, 0.1, 1 );
                this.webcamera.camera.position.set(0.0, 0.0, 0.0);
                this.scene.add(this.webcamera.camera);
                this.webcamera.camera.getWorldPosition();
                this.webcamera.camera.updateProjectionMatrix();
                this.helper = new THREE.CameraHelper( this.webcamera.camera );
                this.scene.add( this.helper );

                this.webcamTexture = new THREE.VideoTexture( this.video );
                this.webcamMat = new THREE.MeshBasicMaterial({ map: this.webcamTexture, side: THREE.DoubleSide }); //
                this.webcamQuad = new THREE.Mesh(new THREE.PlaneBufferGeometry(1, 1), this.webcamMat);
                this.webcamQuad.position.set(0.0, 0.0, -0.2);
                this.webcamQuad.scale.set(this.webcamera.g.width/this.webcamera.g.height * 0.23, 0.23, 1);
                this.scene.add(this.webcamQuad);
                
                if(!this.useFaceLandmarking){
                    let lineMat = new THREE.LineBasicMaterial({ color: 0xffffff });
                    
                    this.linePoints = [];
                    for(let i = 0; i < 33; i++){
                        this.linePoints.push( new THREE.Vector3( 0, 0, 0 ) );
                        this.linePoints.push( new THREE.Vector3( Math.random()-0.5, Math.random()-0.5, -10.1 ) );
                        this.rawPoints.push( new THREE.Vector3( 0, 0, 0 ) );
                        this.filteredPoints.push( new THREE.Vector3( 0, 0, 0 ) );
                    }
                    
                    this.linegeometry = new THREE.BufferGeometry().setFromPoints( this.linePoints );
                    
                    let line = new THREE.LineSegments( this.linegeometry, lineMat );
                    //this.scene.add( line );
                }

                // Just do this once to initialize camera
                this.renderer.render(this.scene, this.webcamera.camera);
            });
        });
    }

    async createLandmarker() {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        if (this.useFaceLandmarking){
            return await FaceLandmarker.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                    delegate: "GPU"
                },
                outputFaceBlendshapes: false,
                runningMode: "VIDEO",
                numFaces: 1
            });
        }else{
            return await PoseLandmarker.createFromOptions(filesetResolver, {
                baseOptions: {
                  modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task`,
                  delegate: "GPU"
                },
                outputFaceBlendshapes: false,
                runningMode: "VIDEO",
                numFaces: 1
            });
        }
    }

    async detectLandmarks() {
        if (this.video && this.videoIsPlaying && this.lastVideoTime !== this.video.currentTime) {
            this.lastVideoTime = this.video.currentTime;
            this.onResults(this.faceMesh.detectForVideo(this.video, performance.now()));
        }
    }

    onResults(results) {
        this.landmarks.count = 0;

        this.webcamera.camera.fov = this.params.FoV;
        this.webcamera.camera.updateProjectionMatrix();
        this.helper.update();

        if (this.useFaceLandmarking){
            if (!results.faceLandmarks) { return; }

            // Find the Average
            this.centroid.set(0, 0, 0);
            let pos = results.faceLandmarks[0];
            if (!pos) { return; }
            for (let i = 0; i < pos.length; i++){
                this.transformPoint(pos[i], this.vec);
                this.centroid.add(this.vec);
            }
            this.centroid.divideScalar(pos.length);

            // Find the Sum Distance from the Average (Scale)
            let scale = 0;
            for (let i = 0; i < pos.length; i++){
                this.transformPoint(pos[i], this.vec);
                scale += this.vec3.copy(this.centroid).sub(this.vec).length();
            }

            // Divide position by scale to get 3D position
            for (let i = 0; i < pos.length; i++){
                this.transformPoint(pos[i], this.vec);
                this.vec.multiplyScalar(1.0/(scale * this.params.DepthScalar));
                
                //let dist = this.vec.length();
                let z = this.vec.z;

                this.vec.set((pos[i].x * 2.0) - 1.0, (pos[i].y * -2.0) + 1.0, -1.0).unproject(this.webcamera.camera);

                //this.vec.setLength(dist);
                this.vec.divideScalar(this.vec.z / z);

                this.landmarks.setMatrixAt(i, this.mat.compose(this.vec, this.quat, this.vec6.set(0.00025, 0.00025, 0.00025)));
            }

            this.landmarks.count = pos.length;
            this.landmarks.instanceMatrix.needsUpdate = true;
        }else{
            // Find the Averages of the Landmarks
            this.worldCentroid = new THREE.Vector3();
            this.cameraCentroid = new THREE.Vector3();

            this.temp2 = new THREE.Vector3();

            let pos  = results.landmarks[0];
            let wpos = results.worldLandmarks[0];
            let previousCost = -1;
            let currentCost  =  0;
            if (!pos) { return; }

            for (let iter = 0; iter < 1000; iter++){
                currentCost = 0;

                for (let i = 0; i < pos.length; i++){
                    // Set the sphere position
                    this.vec.set(wpos[i].x* 0.1, -wpos[i].y* 0.1, -wpos[i].z * 0.1).add(this.offset);
                    this.worldCentroid.add(this.vec);

                    // Set the end of the line segment
                    this.linePoints[(i*2)+1].set((pos[i].x * 2.0) - 1.0, (pos[i].y * -2.0) + 1.0, -1.0).unproject(this.webcamera.camera).normalize();

                    // Project the world point onto the line segment
                    this.temp2.copy(this.vec).projectOnVector(this.linePoints[(i*2)+1]);
                    this.cameraCentroid.add(this.temp2);
                    this.rawPoints[i].copy(this.temp2);

                    // Calculate the cost
                    currentCost += this.temp2.sub(this.vec).length();
                }
                this.linegeometry.setFromPoints(this.linePoints);

                this. worldCentroid.divideScalar(pos.length);
                this.cameraCentroid.divideScalar(pos.length);

                let multiplier = 1.0;//(iter > 0 && (previousCost - currentCost) > 0.00001) ? (previousCost / (previousCost - currentCost)) * 0.8 : 1.0;
                this.offset.add(this.temp2.subVectors(this.cameraCentroid, this.worldCentroid).multiplyScalar(multiplier));

                //if(iter == 999) { console.log(multiplier, currentCost, previousCost); }
                previousCost = currentCost;

            }
            //console.log(currentCost);

            // Use Point Filtering
            for (let i = 0; i < pos.length; i++){
                // Lerp towards destination
                //this.temp2.subVectors(this.rawPoints[i], this.filteredPoints[i]);
                //this.filteredPoints[i].add(this.temp2.multiplyScalar(0.1));

                // Clamp the distance
                this.temp2.subVectors(this.rawPoints[i], this.filteredPoints[i]);
                this.temp2.setLength(Math.max(this.temp2.length() - 0.00125, 0.0));
                this.filteredPoints[i].add(this.temp2);
                this.landmarks.setMatrixAt(i, this.mat.compose(this.filteredPoints[i], this.quat, this.vec6.set( 0.0025, 0.0025, 0.0025)));
            }

            this.landmarks.count = pos.length;
            this.landmarks.instanceMatrix.needsUpdate = true;
        }
    }

    /** Transform from camera UV space (?) to sort of local 3D space? */
    transformPoint(point, vectorToFill) {
        vectorToFill.set((point.x * 2.0) - 1.0, (point.y * -2.0) + 1.0, -1.0).unproject(this.webcamera.camera);
        if(this.useFaceLandmarking){ vectorToFill.z -= (point.z * this.params.DepthScalar); }
    }

    /** Update the camera and render the scene */
    update() {
        this.detectLandmarks();

        // Update the Orbit Controls
        this.controls.update();

        // Render the scene
        this.renderer.render(this.scene, this.camera);

        this.stats.update();
    }

    /** **INTERNAL**: Set up a basic world */
    _setupWorld() {
        // Record browser metadata for power saving features...
        this.safari = /(Safari)/g.test( navigator.userAgent ) && ! /(Chrome)/g.test( navigator.userAgent );
        this.mobile = /(Android|iPad|iPhone|iPod|Oculus)/g.test(navigator.userAgent) || this.safari;

        // app container div
        this.container = document.getElementById('appbody');
        document.body.appendChild(this.container);
        
        // camera and world
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color( 0xeeeeee );
        //this.scene.fog = new THREE.Fog(0xffffff, 0.5, 1.3);

        this.cameraWorldPosition = new THREE.Vector3(1,1,1);
        this.cameraWorldScale    = new THREE.Vector3(1,1,1);
        this.camera = new THREE.PerspectiveCamera( 60, window.innerWidth / window.innerHeight, 0.0001, 1000 );
        this.camera.position.set( 0, 0.0, 0.0);//0.1, 0.2, 0.3 );
        this.camera.layers.enableAll();
        this.cameraParent = new THREE.Group();
        this.cameraParent.add(this.camera);
        this.scene.add(this.cameraParent);
        this.camera.getWorldPosition(this.cameraWorldPosition);

        // ground
        this.mesh = new THREE.Mesh(new THREE.PlaneBufferGeometry(2, 2),
                                   new THREE.MeshStandardMaterial({ color: 0xffffff, depthWrite: false})); //, opacity: 0 
        this.mesh.rotation.x = - Math.PI / 2;
        //this.mesh.castShadow = true;
        this.mesh.receiveShadow = true;
        this.mesh.frustumCulled = false;
        this.mesh.position.y = -0.1;
        this.scene.add( this.mesh );
        this.grid = new THREE.GridHelper( 2, 20, 0x000000, 0x000000 );
        this.grid.material.opacity = 0.4;
        this.grid.material.transparent = true;
        this.grid.layers.set(2);
        this.grid.frustumCulled = false;
        this.grid.position.y = -0.1;
        this.scene.add(this.grid);
        
        // light
        this.light = new THREE.HemisphereLight( 0xffffff, 0x444444, 0.5 );
        this.light.position.set( 0, 0.0, 0 );
        this.scene.add( this.light );
        this.lightParent = new THREE.Group();
        this.lightTarget = new THREE.Group();
        this.lightParent.frustumCulled = false;
        this.lightParent.add(this.lightTarget);
        this.light = new THREE.DirectionalLight( 0xffffff );
        this.light.position.set( 0, 20, 10);
        this.light.castShadow = !this.mobile;
        this.light.frustumCulled = false;
        this.light.shadow.frustumCulled = false;
        this.light.shadow.camera.frustumCulled = false;
        this.light.shadow.camera.top    =   1;
        this.light.shadow.camera.bottom = - 1;
        this.light.shadow.camera.left   = - 1;
        this.light.shadow.camera.right  =   1;
        //this.light.shadow.autoUpdate = false;
        this.light.target = this.lightTarget;
        this.lightParent.add(this.light);
        this.scene.add( this.lightParent );
        //this.scene.add( new THREE.CameraHelper( this.light.shadow.camera ) );

        // renderer
        this.renderer = new THREE.WebGLRenderer( { antialias: true } );
        this.renderer.setPixelRatio( window.devicePixelRatio );
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);
        this.renderer.setAnimationLoop( this.update.bind(this) );
        
        // orbit controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.10;
        this.controls.screenSpacePanning = true;
        this.controls.target.set( 0, 0.0, -0.1 );
        this.controls.update();
        window.addEventListener('resize', this._onWindowResize.bind(this), false);
        window.addEventListener('orientationchange', this._onWindowResize.bind(this), false);
        this._onWindowResize();

        // raycaster
        this.raycaster = new THREE.Raycaster();
        this.raycaster.layers.set(0);

        // stats
        this.stats = new Stats();
        this.container.appendChild(this.stats.dom);

        // Temp variables to reduce allocations
        this.mat  = new THREE.Matrix4();
        this.vec = new THREE.Vector3();
        this.centroid = new THREE.Vector3();
        this.vec3 = new THREE.Vector3();
        this.vec4 = new THREE.Vector3();
        this.vec5 = new THREE.Vector3();
        this.vec6 = new THREE.Vector3();
        this.zVec = new THREE.Vector3(0, 0, 1);
        this.quat = new THREE.Quaternion().identity();
        this.color = new THREE.Color();
    }

    /** **INTERNAL**: This function recalculates the viewport based on the new window size. */
    _onWindowResize() {
        let rect = this.container.getBoundingClientRect();
        let width = rect.width, height = window.innerHeight - rect.y;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

}

new World();