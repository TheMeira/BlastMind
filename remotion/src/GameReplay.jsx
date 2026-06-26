import { useCurrentFrame, interpolate, spring, useVideoConfig } from "remotion";
import { useMemo } from "react";
import { Board } from "./Board";

const FRAMES_PER_STATE = 6;
const GAME_OVER_HOLD   = 90;

export const calculateMetadata = ({ props }) => ({
    durationInFrames: props.frames.length * FRAMES_PER_STATE + GAME_OVER_HOLD,
});

const PIECE_COLORS = {
    P01: "#00e5ff",
    P02: "#448aff", P03: "#448aff", P04: "#448aff", P05: "#448aff",
    P06: "#448aff", P07: "#448aff", P08: "#448aff", P09: "#448aff",
    P10: "#ff6d00", P11: "#ff6d00", P12: "#ff6d00", P13: "#ff6d00",
    P14: "#ffea00", P15: "#ffea00", P16: "#ffea00", P17: "#ffea00",
    P18: "#ff1744", P19: "#ff1744", P20: "#ff1744", P21: "#ff1744",
    P22: "#00e676", P23: "#00e676", P24: "#00e676", P25: "#00e676",
    P26: "#00e676", P27: "#00e676", P28: "#00e676", P29: "#00e676",
    P30: "#e040fb", P31: "#e040fb", P32: "#e040fb", P33: "#e040fb",
    P34: "#7c4dff", P35: "#7c4dff", P36: "#7c4dff", P37: "#7c4dff",
};

function buildColorGrids(frames) {
    const grids = [];
    let grid = Array.from({ length: 8 }, () => Array(8).fill(null));

    for (let i = 0; i < frames.length; i++) {
        const f = frames[i];
        if (f.last_piece_id && i > 0) {
            const prev = frames[i - 1].board;
            const next = grid.map(r => [...r]);
            const color = PIECE_COLORS[f.last_piece_id] || "#448aff";
            for (let r = 0; r < 8; r++) {
                for (let c = 0; c < 8; c++) {
                    if (f.board[r][c] === 0)                             next[r][c] = null;
                    else if (f.board[r][c] === 1 && prev[r][c] === 0)   next[r][c] = color;
                }
            }
            grid = next;
        }
        grids.push(grid.map(r => [...r]));
    }
    return grids;
}

export const GameReplay = ({ frames, agent, finalScore, seed }) => {
    const frame      = useCurrentFrame();
    const { fps }    = useVideoConfig();
    const colorGrids = useMemo(() => buildColorGrids(frames), [frames]);

    const gameplayFrames = frames.length * FRAMES_PER_STATE;
    const isGameOver     = frame >= gameplayFrames;
    const stateIdx       = Math.min(Math.floor(frame / FRAMES_PER_STATE), frames.length - 1);
    const currentState   = frames[stateIdx];
    const colorGrid      = colorGrids[stateIdx];

    const gameOverProgress = isGameOver
        ? spring({ frame: frame - gameplayFrames, fps, config: { damping: 14 } })
        : 0;

    const agentLabel = agent.charAt(0).toUpperCase() + agent.slice(1) + " Agent";

    return (
        <div style={{
            width: "100%", height: "100%",
            background: "#070714",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            fontFamily: "'Courier New', monospace",
            position: "relative",
            overflow: "hidden",
        }}>
            <div style={{
                position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
                background: "radial-gradient(ellipse at 50% 0%, #0a1a3a 0%, #070714 70%)",
            }} />

            <div style={{ position: "relative", display: "flex", flexDirection: "column", alignItems: "center", gap: 24 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 40 }}>
                    <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: 18, color: "#2a5a7a", letterSpacing: 3, textTransform: "uppercase" }}>
                            {agentLabel}
                        </div>
                        <div style={{ fontSize: 14, color: "#1a3050", marginTop: 4 }}>
                            seed {seed}
                        </div>
                    </div>

                    <Board board={currentState.board} colorGrid={colorGrid} />

                    <div style={{ textAlign: "left" }}>
                        <div style={{ fontSize: 16, color: "#2a5a7a", letterSpacing: 2, textTransform: "uppercase" }}>
                            Score
                        </div>
                        <div style={{
                            fontSize: 48, fontWeight: "bold",
                            color: "#00d4ff",
                            textShadow: "0 0 30px #00aaff",
                            lineHeight: 1.1,
                        }}>
                            {currentState.score}
                        </div>
                        {currentState.combo_count > 1 && (
                            <div style={{ fontSize: 20, color: "#00e5ff", marginTop: 4 }}>
                                ×{currentState.combo_count} combo
                            </div>
                        )}
                    </div>
                </div>

                <div style={{ fontSize: 13, color: "#1a3050", letterSpacing: 2 }}>
                    BLASTMIND  ·  COMP4026  ·  University of Nottingham
                </div>
            </div>

            {isGameOver && (
                <div style={{
                    position: "absolute", inset: 0,
                    background: `rgba(7,7,20,${interpolate(gameOverProgress, [0, 1], [0, 0.88])})`,
                    display: "flex", flexDirection: "column",
                    alignItems: "center", justifyContent: "center",
                    gap: 20,
                    transform: `scale(${interpolate(gameOverProgress, [0, 1], [1.08, 1])})`,
                }}>
                    <div style={{
                        fontSize: 80, fontWeight: "bold",
                        color: "#ff1744",
                        textShadow: "0 0 60px #ff1744",
                        opacity: gameOverProgress,
                    }}>
                        GAME OVER
                    </div>
                    <div style={{
                        fontSize: 28, color: "#2a5a7a",
                        opacity: gameOverProgress,
                    }}>
                        Final Score
                    </div>
                    <div style={{
                        fontSize: 96, fontWeight: "bold",
                        color: "#00d4ff",
                        textShadow: "0 0 50px #00aaff",
                        opacity: gameOverProgress,
                        lineHeight: 1,
                    }}>
                        {finalScore}
                    </div>
                    <div style={{
                        fontSize: 18, color: "#1a4060",
                        opacity: gameOverProgress,
                        marginTop: 8,
                    }}>
                        {agentLabel}  ·  seed {seed}
                    </div>
                </div>
            )}
        </div>
    );
};
