import * as THREE from '../node_modules/three/build/three.module.js';
import Stats from '../node_modules/three/examples/jsm/libs/stats.module.js';
import { GUI } from '../node_modules/three/examples/jsm/libs/dat.gui.module.js';
import { OrbitControls } from '../node_modules/three/examples/jsm/controls/OrbitControls.js';

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
        this.sphereGeometry = new THREE.SphereGeometry(1.0, 12, 12);
        this.solidMat = new THREE.MeshPhongMaterial({ wireframe: false });
        this.landmarks = new THREE.InstancedMesh(this.sphereGeometry, this.solidMat, 468);
        this.landmarks.receiveShadow = true;
        this.landmarks.castShadow = true;
        this.scene.add(this.landmarks);

        // Construct the Face Tracking Pipeline
        this.setupFaceTracking();
    }

    async setupFaceTracking() {
        this.faceMesh = new FaceMesh({locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.1/${file}`; //../node_modules/@mediapipe/face_mesh/
        }});
        this.faceMesh.onResults(this.onResults.bind(this));
        
        // Instantiate a camera. We'll feed each frame we receive into the solution.
        let videoElement = document.getElementsByClassName('input_video')[0];
        this.webcamera = new Camera(videoElement, {
          onFrame: async () => {
            await this.faceMesh.send({image: videoElement});
          }, width: 1280, height: 720
        });
        this.webcamera.start();

        this.webcamera.camera = new THREE.PerspectiveCamera( 78, this.webcamera.g.width / this.webcamera.g.height, 0.1, 1 );
        this.webcamera.camera.position.set(0.0, 0.1, 0.2);
        this.scene.add(this.webcamera.camera);
        this.webcamera.camera.getWorldPosition();
        this.webcamera.camera.updateProjectionMatrix();
        this.helper = new THREE.CameraHelper( this.webcamera.camera );
        this.scene.add( this.helper );

        // Just do this once to initialize camera
        this.renderer.render(this.scene, this.webcamera.camera);
    }

    onResults(results) {
        if (!results.multiFaceLandmarks) { return; }

        this.webcamera.camera.fov = this.params.FoV;
        this.webcamera.camera.updateProjectionMatrix();
        this.helper.update();

        // Find the Average
        this.vec2.set(0, 0, 0);
        let pos = results.multiFaceLandmarks[0];
        for (let i = 0; i < pos.length; i++){
            this.transformPoint(pos[i], this.vec);
            this.vec2.add(this.vec);
        }
        this.vec2.divideScalar(pos.length);

        // Find the Sum Distance from the Average (Scale)
        let scale = 0;
        for (let i = 0; i < pos.length; i++){
            this.transformPoint(pos[i], this.vec);
            scale += this.vec3.copy(this.vec2).sub(this.vec).length();
        }

        // Divide position by scale to get 3D position
        for (let i = 0; i < pos.length; i++){
            this.transformPoint(pos[i], this.vec);
            this.vec.sub(this.webcamera.camera.position).multiplyScalar(5.0/scale).add(this.webcamera.camera.position);
            this.landmarks.setMatrixAt(i, this.mat.compose(this.vec, this.quat, this.vec6.set(0.001, 0.001, 0.001)));
            //this.landmarks.setColorAt(i, this.color.setRGB(i, Math.random(), Math.random()));
        }

        this.landmarks.count = pos.length;
        this.landmarks.instanceMatrix.needsUpdate = true;
        //this.landmarks.instanceColor.needsUpdate = true;
    }

    /** Transform from camera UV space (?) to sort of local 3D space? */
    transformPoint(point, vectorToFill) {
        vectorToFill.set((point.x * 2.0) - 1.0, (point.y * -2.0) + 1.0, -1.0).unproject(this.webcamera.camera);
        vectorToFill.z -= (point.z * this.params.DepthScalar);
    }

    /** Update the camera and render the scene */
    update() {
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
        this.camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.0001, 1000 );
        this.camera.position.set( 0.1, 0.2, 0.3 );
        this.camera.layers.enableAll();
        this.cameraParent = new THREE.Group();
        this.cameraParent.add(this.camera);
        this.scene.add(this.cameraParent);
        this.camera.getWorldPosition(this.cameraWorldPosition);

        // ground
        this.mesh = new THREE.Mesh(new THREE.PlaneBufferGeometry(2, 2),
                                   new THREE.MeshStandardMaterial({ color: 0xaaaaaa, depthWrite: false})); //, opacity: 0 
        this.mesh.rotation.x = - Math.PI / 2;
        //this.mesh.castShadow = true;
        this.mesh.receiveShadow = true;
        this.mesh.frustumCulled = false;
        this.scene.add( this.mesh );
        this.grid = new THREE.GridHelper( 2, 20, 0x000000, 0x000000 );
        this.grid.material.opacity = 0.4;
        this.grid.material.transparent = true;
        this.grid.layers.set(2);
        this.grid.frustumCulled = false;
        this.scene.add(this.grid);
        
        // light
        this.light = new THREE.HemisphereLight( 0xffffff, 0x444444, 0.5 );
        this.light.position.set( 0, 0.2, 0 );
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
        this.controls.target.set( 0, 0.1, 0 );
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
        this.vec2 = new THREE.Vector3();
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