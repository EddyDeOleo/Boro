Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
# Use a natural, expressive voice (Microsoft Zira Desktop)
$synth.SelectVoice('Microsoft Zira Desktop')

# Define narration texts per scene
$scenes = @{
    'scene1.wav' = @(
        'In autonomous robotics, movement is often mistaken for success.',
        'A robot reaches its target coordinates, but did the object stay in the gripper? Or did it slip unnoticed?',
        'Blindly trusting coordinate execution hides critical failures.'
    ) -join ' '
    'scene2.wav' = @(
        'Introducing Boro, a robotic execution intelligence platform built for the Meta Global AI Developer Hackathon.',
        'Boro bridges the gap between natural language instructions and physical reality, converting visual streams into auditable verification evidence.'
    ) -join ' '
    'scene3.wav' = @(
        'The pipeline begins with Meta Muse Spark. Rather than relying on rigid rules, it interprets natural language instructions into structured task definitions, assigning semantic roles and establishing criteria with explicit mandatory requirements.',
        'Muse Spark converts unstructured prompts into formal TaskRoles and VerificationCriteria.'
    ) -join ' '
    'scene4.wav' = @(
        'Next, Meta SAM 3.1 performs spatio-temporal segmentation. From initial bounding boxes, SAM tracks the gripper and object across every frame.',
        'Boro extracts two-dimensional centroid trajectories, providing visual evidence without assuming unmeasured physical forces.'
    ) -join ' '
    'scene5.wav' = @(
        'Here is episode zero. The robot cleanly grasps the cube, transports it across the table, releases it inside the container, and retracts.',
        'Every mandatory criterion is satisfied across all three hundred and one frames, earning an explainable verdict of Verified.'
    ) -join ' '
    'scene6.wav' = @(
        'Now examine episode twenty‑five. In this critical 1.67‑second segment, the cube slips prematurely.',
        'Boro detects relative separation, but because the clip ends before confirming placement, VerdictEngine issues an explainable verdict: Insufficient Evidence.'
    ) -join ' '
    'scene7.wav' = @(
        'Through Boro''s modern interface, roboticists inspect temporal timelines, review confidence scores, and read transparent limitations.',
        'Boro explicitly documents boundary assumptions, such as two‑dimensional constraints, ensuring human operators can audit every verification.'
    ) -join ' '
    'scene8.wav' = @(
        'Built for the Meta Global AI Developer Hackathon 2026, Boro delivers honest, auditable intelligence to robotic workflows.',
        'Because a robot''s movement is not proof of success. Evidence is.'
    ) -join ' '
}

# Generate each WAV file
foreach ($file in $scenes.Keys) {
    $text = $scenes[$file]
    $outputPath = Join-Path -Path (Resolve-Path 'C:\Users\User\source\repos\Boro\demo-video\public\audio') -ChildPath $file
    Write-Host "Generating $file ..."
    $synth.SetOutputToWaveFile($outputPath)
    $synth.Speak($text)
    $synth.SetOutputToDefaultAudioDevice()
}

Write-Host 'Narration generation complete.'
