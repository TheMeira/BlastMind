import { Composition } from "remotion";
import { GameReplay, calculateMetadata } from "./GameReplay";
import replayData from "../../replays/random.json";

export const Root = () => {
    return (
        <Composition
            id="GameReplay"
            component={GameReplay}
            calculateMetadata={calculateMetadata}
            fps={30}
            width={1280}
            height={720}
            defaultProps={{
                agent:      replayData.agent,
                finalScore: replayData.final_score,
                seed:       replayData.seed,
                frames:     replayData.frames,
            }}
        />
    );
};
