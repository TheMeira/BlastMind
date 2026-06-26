const CELL  = 56;
const BOARD = 8 * CELL;
const PAD   = 3;
const R     = 5;

function lighten(hex, amount) {
    const num = parseInt(hex.replace("#", ""), 16);
    const r   = Math.min(255, (num >> 16) + amount);
    const g   = Math.min(255, ((num >> 8) & 0xff) + amount);
    const b   = Math.min(255, (num & 0xff) + amount);
    return `rgb(${r},${g},${b})`;
}

export const Board = ({ board, colorGrid }) => {
    return (
        <div style={{
            width:  BOARD + 12,
            height: BOARD + 12,
            background: "#0a0a1e",
            borderRadius: 10,
            padding: 6,
            boxShadow: "0 0 40px rgba(0,100,200,0.15), inset 0 0 0 1px #0d2244",
            display: "grid",
            gridTemplateColumns: `repeat(8, ${CELL}px)`,
            gridTemplateRows:    `repeat(8, ${CELL}px)`,
        }}>
            {board.map((row, r) =>
                row.map((cell, c) => {
                    const color = cell === 1 ? (colorGrid[r][c] || "#448aff") : null;
                    return (
                        <div key={`${r}-${c}`} style={{
                            width:  CELL - PAD * 2,
                            height: CELL - PAD * 2,
                            margin: PAD,
                            borderRadius: R,
                            background: color ? color : "#181830",
                            boxShadow: color
                                ? `inset 0 ${Math.floor(CELL * 0.12)}px 0 0 ${lighten(color, 40)}`
                                : "none",
                            transition: "background 0.05s",
                        }} />
                    );
                })
            )}
        </div>
    );
};
