import { Application, Entity } from '@playcanvas/react'
import { Camera, Render } from '@playcanvas/react/components'
import { OrbitControls } from '@playcanvas/react/scripts'
// import your state from wherever you keep it
import { useStore } from '@nanostores/react'
import { theme } from '@/stores/sceneStore'
import { useModel } from '@playcanvas/react/hooks'

function Scene() {
  const currentTheme = useStore(theme)
// 1. Load BOTH models at the start, just like your prototype
  const { 
    asset: lightModelAsset, 
    loading: loadingLight, 
    error: errorLight 
  } = useModel('/models/cat.glb')
  
  const { 
    asset: darkModelAsset, 
    loading: loadingDark, 
    error: errorDark 
  } = useModel('/models/your-model-dark.glb')

  // 2. Show a loading state until BOTH are ready
  if (loadingLight || loadingDark) return <div>Loading models...</div>
  if (errorLight) return <div>Error loading light model: {String(errorLight)}</div>
  if (errorDark) return <div>Error loading dark model: {String(errorDark)}</div>
  if (!lightModelAsset || !darkModelAsset) return null

  // 3. Just pick which asset to show.
  // This is the *exact same logic* as your prototype's theme toggle.
  const assetToRender = currentTheme === 'dark' ? darkModelAsset : lightModelAsset

  // const { asset, loading, error } = useModel(modelSrc)
/* 
  if (loading) return <div>Loading model...</div>
  if (error) return <div>Error loading model: {String(error)}</div>
  if (!asset) return null */

  const mouseConfig =  typeof window !== 'undefined' && (window as any).pcApp && (window as any).pcApp.mouse 
  ? {
                buttons: {
                  rotate: 1,   // left button
                  pan: 2,      // right button
                  zoom: 4      // middle button
                }
              } as any
  : { enabled: false };

  return (
    <>
      <Entity name="camera" position={[4, 3, 4]}>
        <Camera fov={45} />
        <OrbitControls 
            inertiaFactor={0.07} 
            mouse={
              mouseConfig
            }
        />
      </Entity>
      {/* Example: Dynamic model switching */}
      <Entity>
        <Render type="asset" asset={assetToRender} />
      </Entity>
    </>
  )
}

export default function ImmersiveScene() {
  return (
    <Application
      fillMode="FILL_WINDOW"
      resolutionMode="AUTO"
      usePhysics={true}
      // keyboard, mouse, etc. are automatically enabled
    >
      <Scene />
    </Application>
  )
}
