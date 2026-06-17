const BOARD_N = 8;

const PIECE_COLORS = {
    P01: 0x00e5ff,
    P02: 0x448aff, P03: 0x448aff,
    P04: 0x448aff, P05: 0x448aff,
    P06: 0x448aff, P07: 0x448aff,
    P08: 0x448aff, P09: 0x448aff,
    P10: 0xff6d00, P11: 0xff6d00, P12: 0xff6d00, P13: 0xff6d00,
    P14: 0xffea00, P15: 0xffea00, P16: 0xffea00, P17: 0xffea00,
    P18: 0xff1744, P19: 0xff1744, P20: 0xff1744, P21: 0xff1744,
    P22: 0x00e676, P23: 0x00e676, P24: 0x00e676, P25: 0x00e676,
    P26: 0x00e676, P27: 0x00e676, P28: 0x00e676, P29: 0x00e676,
    P30: 0xe040fb, P31: 0xe040fb, P32: 0xe040fb, P33: 0xe040fb,
    P34: 0x7c4dff, P35: 0x7c4dff, P36: 0x7c4dff, P37: 0x7c4dff,
};

class GameScene extends Phaser.Scene {
    constructor() {
        super({ key: 'GameScene' });
    }

    init(data) {
        this.mode = (data && data.mode) || 'human';
        this.aiSpeed = (data && data.speed) || 1.0;
        this.aiAgent = (data && data.agent) || 'random';
        this.resume = (data && data.resume) || false;
        this.gameState = null;
        this.selectedIdx = null;
        this.ws = null;
        this.pieceZones = [];
        this.bgBlocks = [];
        this.colorGrid = Array.from({ length: BOARD_N }, () => Array(BOARD_N).fill(null));
    }

    preload() {
        if (!this.cache.audio.exists('click')) {
            this.load.audio('click', '/static/audio/click.mp3');
        }
        if (!this.cache.audio.exists('line-clear')) {
            this.load.audio('line-clear', '/static/audio/line-clear.mp3');
        }
    }

    create() {
        this.cameras.main.setBackgroundColor('#070714');

        const W = this.scale.width;
        const H = this.scale.height;

        const cellByH = Math.floor((H * 0.60) / BOARD_N);
        const cellByW = Math.floor((W * 0.54) / BOARD_N);
        this.CS = Math.min(104, cellByH, cellByW);
        this.BP = BOARD_N * this.CS;
        this.BX = Math.floor((W - this.BP) / 2);
        this.BY = Math.floor(H * 0.09);

        const slotH = Math.floor(this.CS * 2.5);
        this.SLOT = slotH;
        this.PANEL_Y = this.BY + this.BP + Math.floor(this.CS * 0.28);
        this.CX = W / 2;
        this.W = W;
        this.H = H;

        this.spawnBackground(W, H);
        this.drawCenterPanel(W, H);

        this.boardGfx = this.add.graphics();
        this.ghostGfx = this.add.graphics();
        this.panelGfx = this.add.graphics();

        const vpad = Math.floor(H * 0.02);
        const scoreFS = Math.floor(Math.min(H * 0.040, 42));
        const scoreCY = Math.floor((vpad + this.BY) / 2);
        this.scoreLbl = this.add.text(W / 2, scoreCY, 'Score: 0', {
            fontFamily: 'Orbitron, Arial',
            fontSize: scoreFS + 'px',
            color: '#ffffff',
            fontStyle: 'bold',
        }).setOrigin(0.5, 0.5);

        this.comboLbl = this.add.text(W / 2, scoreCY + Math.floor(scoreFS * 0.55) + 4, '', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(scoreFS * 0.6) + 'px',
            color: '#ffd740',
        }).setOrigin(0.5, 0);

        const panelBottom = this.PANEL_Y + this.SLOT;
        const hintCY = Math.floor((panelBottom + (H - vpad)) / 2);
        this.hintLbl = this.add.text(W / 2, hintCY, '', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(H * 0.018) + 'px',
            color: '#2a4a6a',
        }).setOrigin(0.5, 0.5);

        const menuBtn = this.add.text(16, 16, '← Menu', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(H * 0.022) + 'px',
            color: '#1a3a5a',
        }).setInteractive({ useHandCursor: true });
        menuBtn.on('pointerover', () => menuBtn.setStyle({ color: '#00d4ff' }));
        menuBtn.on('pointerout', () => menuBtn.setStyle({ color: '#1a3a5a' }));
        menuBtn.on('pointerdown', () => {
            try { this.sound.play('click', { volume: getSFXVolume() }); } catch (e) {}
            if (this.ws) this.ws.close();
            this.scene.start('MenuScene');
        });

        this.drawEmptyBoard();

        if (this.mode === 'human') {
            this.setupInput();
            this.hintLbl.setText('Click a piece below, then click the board to place it');
            if (this.resume) {
                this.resumeHumanGame();
            } else {
                this.startHumanGame();
            }
        } else {
            this.hintLbl.setText('AI Watch Mode · ' + this.aiAgent + ' · ' + this.aiSpeed + 'x speed');
            this.startAIWatch();
        }

        this.scale.on('resize', this._onResize, this);
        this.events.once('shutdown', () => this.scale.off('resize', this._onResize, this));
        this.events.on('shutdown', this.shutdown, this);
    }

    _onResize() {
        clearTimeout(this._resizeTimer);
        this._resizeTimer = setTimeout(() => {
            if (this.ws) { this.ws.close(); this.ws = null; }
            this.scene.restart({ mode: this.mode, speed: this.aiSpeed, agent: this.aiAgent, resume: this.mode === 'human' });
        }, 200);
    }

    spawnBackground(W, H) {
        const colors = [0x448aff, 0xff6d00, 0xffea00, 0xff1744, 0x00e676, 0xe040fb, 0x7c4dff, 0x00e5ff];
        const { BX, BP } = this;
        const zones = [
            { x: 0, w: BX },
            { x: BX + BP, w: W - (BX + BP) },
        ];
        const sideArea = (zones[0].w + zones[1].w) * H;
        const count = Math.max(6, Math.floor(sideArea / 18000));

        for (let i = 0; i < count; i++) {
            const zone = zones[i % 2];
            if (zone.w < 20) continue;
            const maxSize = Math.floor(Math.min(zone.w * 0.55, 70));
            const size = Phaser.Math.Between(16, Math.max(17, maxSize));
            const half = Math.ceil(size / 2);
            const spawnW = Math.max(0, zone.w - 2 * half);
            const x = zone.x + half + Phaser.Math.Between(0, spawnW);
            const y = Phaser.Math.Between(-80, H + 80);
            const color = colors[i % colors.length];
            const alpha = Phaser.Math.FloatBetween(0.06, 0.15);
            const rect = this.add.rectangle(x, y, size, size, color, alpha);
            rect.rotation = Phaser.Math.FloatBetween(0, Math.PI * 2);
            this.bgBlocks.push({
                rect,
                vy: Phaser.Math.FloatBetween(-0.8, -0.2),
                rs: Phaser.Math.FloatBetween(-0.004, 0.004),
                H,
                zoneX: zone.x,
                half,
                spawnW,
            });
        }
    }

    update() {
        this.bgBlocks.forEach(b => {
            b.rect.y += b.vy;
            b.rect.rotation += b.rs;
            if (b.rect.y < -200) {
                b.rect.y = b.H + 100;
                b.rect.x = b.zoneX + b.half + Phaser.Math.Between(0, b.spawnW);
            }
        });
    }

    drawCenterPanel(W, H) {
        const hpad = Math.floor(this.CS * 0.6);
        const vpad = Math.floor(H * 0.02);
        const x = this.BX - hpad;
        const y = vpad;
        const w = this.BP + hpad * 2;
        const h = H - vpad * 2;
        const g = this.add.graphics();
        const r = Math.floor(this.CS * 0.4);
        g.fillStyle(0x070714, 1);
        g.fillRoundedRect(x, y, w, h, r);
        g.lineStyle(2, 0x0d2244, 1);
        g.strokeRoundedRect(x, y, w, h, r);
    }

    drawEmptyBoard() {
        const g = this.boardGfx;
        const { BX, BY, BP, CS } = this;
        g.clear();
        g.fillStyle(0x0a0a1e);
        g.fillRoundedRect(BX - 6, BY - 6, BP + 12, BP + 12, 8);
        for (let r = 0; r < BOARD_N; r++) {
            for (let c = 0; c < BOARD_N; c++) {
                g.fillStyle(0x181830);
                g.fillRoundedRect(BX + c * CS + 3, BY + r * CS + 3, CS - 6, CS - 6, 6);
            }
        }
    }

    renderBoard() {
        const g = this.boardGfx;
        const { BX, BY, BP, CS } = this;
        g.clear();
        g.fillStyle(0x0a0a1e);
        g.fillRoundedRect(BX - 6, BY - 6, BP + 12, BP + 12, 8);
        for (let r = 0; r < BOARD_N; r++) {
            for (let c = 0; c < BOARD_N; c++) {
                const x = BX + c * CS;
                const y = BY + r * CS;
                if (this.gameState.board[r][c] === 1) {
                    const col = this.colorGrid[r][c] || 0x448aff;
                    g.fillStyle(col);
                    g.fillRoundedRect(x + 3, y + 3, CS - 6, CS - 6, 6);
                    g.fillStyle(Phaser.Display.Color.IntegerToColor(col).lighten(30).color);
                    g.fillRoundedRect(x + 6, y + 6, CS - 12, Math.floor(CS * 0.12), 3);
                } else {
                    g.fillStyle(0x181830);
                    g.fillRoundedRect(x + 3, y + 3, CS - 6, CS - 6, 6);
                }
            }
        }
    }

    renderPanel() {
        const g = this.panelGfx;
        g.clear();
        this.pieceZones.forEach(z => z.destroy());
        this.pieceZones = [];

        if (!this.gameState || !this.gameState.pieces.length) return;

        const pieces = this.gameState.pieces;
        const { SLOT, PANEL_Y, CX, CS, W } = this;
        const GAP = Math.floor(CS * 0.30);
        const totalW = pieces.length * SLOT + (pieces.length - 1) * GAP;
        const startX = CX - totalW / 2;
        const R = Math.floor(SLOT * 0.055);

        pieces.forEach((piece, idx) => {
            const sx = startX + idx * (SLOT + GAP);
            const cx = sx + SLOT / 2;
            const cy = PANEL_Y + SLOT / 2;
            const selected = this.selectedIdx === idx;

            g.fillStyle(selected ? 0x0d1e50 : 0x0a0a20);
            g.fillRoundedRect(sx, PANEL_Y, SLOT, SLOT, R);
            if (selected) {
                g.lineStyle(2, 0x00d4ff);
                g.strokeRoundedRect(sx, PANEL_Y, SLOT, SLOT, R);
            }

            const grid = piece.grid;
            const ph = grid.length;
            const pw = grid[0].length;
            const drawArea = SLOT * 0.72;
            const pCell = Math.min(Math.floor(CS * 0.54), Math.floor(drawArea / Math.max(ph, pw)));
            const drawX = cx - (pw * pCell) / 2;
            const drawY = cy - (ph * pCell) / 2;
            const color = PIECE_COLORS[piece.id] || 0x448aff;

            g.fillStyle(color);
            for (let r = 0; r < ph; r++) {
                for (let c = 0; c < pw; c++) {
                    if (grid[r][c] === 1) {
                        g.fillRoundedRect(
                            drawX + c * pCell + 2, drawY + r * pCell + 2,
                            pCell - 4, pCell - 4, 3
                        );
                    }
                }
            }

            if (this.mode === 'human') {
                const zone = this.add.zone(sx, PANEL_Y, SLOT, SLOT).setOrigin(0, 0).setInteractive({ useHandCursor: true });
                zone.on('pointerdown', () => this.selectPiece(idx));
                this.pieceZones.push(zone);
            }
        });
    }

    selectPiece(idx) {
        this.selectedIdx = idx;
        this.ghostGfx.clear();
        this.renderPanel();
    }

    computeAnchor(grid, cursorRow, cursorCol) {
        return {
            row: cursorRow - Math.floor(grid.length / 2),
            col: cursorCol - Math.floor(grid[0].length / 2),
        };
    }

    showGhost(cursorRow, cursorCol) {
        if (this.selectedIdx === null || !this.gameState) return;
        const piece = this.gameState.pieces[this.selectedIdx];
        if (!piece) return;

        const g = this.ghostGfx;
        const { BX, BY, CS } = this;
        g.clear();

        const grid = piece.grid;
        const { row, col } = this.computeAnchor(grid, cursorRow, cursorCol);
        const valid = this.canPlace(grid, row, col);
        const baseColor = PIECE_COLORS[piece.id] || 0x448aff;

        g.fillStyle(valid ? baseColor : 0xff1744, 0.45);
        for (let r = 0; r < grid.length; r++) {
            for (let c = 0; c < grid[0].length; c++) {
                if (grid[r][c] === 1) {
                    const br = row + r;
                    const bc = col + c;
                    if (br >= 0 && br < BOARD_N && bc >= 0 && bc < BOARD_N) {
                        g.fillRoundedRect(BX + bc * CS + 3, BY + br * CS + 3, CS - 6, CS - 6, 6);
                    }
                }
            }
        }
    }

    canPlace(grid, row, col) {
        const ph = grid.length;
        const pw = grid[0].length;
        if (row < 0 || col < 0 || row + ph > BOARD_N || col + pw > BOARD_N) return false;
        for (let r = 0; r < ph; r++) {
            for (let c = 0; c < pw; c++) {
                if (grid[r][c] === 1 && this.gameState.board[row + r][col + c] === 1) return false;
            }
        }
        return true;
    }

    setupInput() {
        const { BX, BY, CS } = this;
        this.input.on('pointermove', (ptr) => {
            if (this.selectedIdx === null) return;
            const col = Math.floor((ptr.x - BX) / CS);
            const row = Math.floor((ptr.y - BY) / CS);
            if (row >= 0 && row < BOARD_N && col >= 0 && col < BOARD_N) {
                this.showGhost(row, col);
            } else {
                this.ghostGfx.clear();
            }
        });

        this.input.on('pointerdown', (ptr) => {
            if (this.selectedIdx === null) return;
            const col = Math.floor((ptr.x - BX) / CS);
            const row = Math.floor((ptr.y - BY) / CS);
            if (row >= 0 && row < BOARD_N && col >= 0 && col < BOARD_N) {
                this.handleBoardClick(row, col);
            }
        });
    }

    updateColorGrid(prevBoard, newBoard, pieceId) {
        const color = PIECE_COLORS[pieceId] || 0x448aff;
        for (let r = 0; r < BOARD_N; r++) {
            for (let c = 0; c < BOARD_N; c++) {
                if (newBoard[r][c] === 0) {
                    this.colorGrid[r][c] = null;
                } else if (newBoard[r][c] === 1 && prevBoard[r][c] === 0) {
                    this.colorGrid[r][c] = color;
                }
            }
        }
    }

    findClearedCells(prevBoard, newBoard) {
        const cells = [];
        for (let r = 0; r < BOARD_N; r++)
            for (let c = 0; c < BOARD_N; c++)
                if (prevBoard[r][c] === 1 && newBoard[r][c] === 0)
                    cells.push({ r, c });
        return cells;
    }

    animatePlacement(prevBoard, newBoard) {
        const { BX, BY, CS } = this;
        for (let r = 0; r < BOARD_N; r++) {
            for (let c = 0; c < BOARD_N; c++) {
                if (prevBoard[r][c] === 0 && newBoard[r][c] === 1) {
                    const flash = this.add.rectangle(
                        BX + c * CS + CS / 2, BY + r * CS + CS / 2,
                        CS - 4, CS - 4, 0xffffff, 0.8
                    );
                    this.tweens.add({
                        targets: flash, alpha: 0, duration: 320, ease: 'Power2',
                        onComplete: () => flash.destroy(),
                    });
                }
            }
        }
    }

    animateClear(cells) {
        const { BX, BY, CS } = this;
        cells.forEach(({ r, c }, i) => {
            const flash = this.add.rectangle(
                BX + c * CS + CS / 2, BY + r * CS + CS / 2,
                CS - 4, CS - 4, 0xffffff, 1
            );
            this.tweens.add({
                targets: flash, alpha: 0, scaleX: 1.6, scaleY: 1.6,
                duration: 520, delay: Math.min(i * 5, 100), ease: 'Expo.Out',
                onComplete: () => flash.destroy(),
            });
        });
    }

    showScorePop(gain, comboCount) {
        if (gain <= 0) return;
        const { BX, BY, BP, H } = this;
        const fs = Math.floor(Math.min(H * 0.055, 56));
        const cx = BX + BP / 2;
        const cy = BY + BP / 2;
        const targetY = BY + BP * 0.25;

        const gainTxt = this.add.text(cx, cy, '+' + gain, {
            fontFamily: 'Orbitron, Arial', fontSize: fs + 'px', fontStyle: 'bold',
            color: '#ffd740',
            shadow: { offsetX: 0, offsetY: 0, color: '#ff8800', blur: 20, fill: true },
        }).setOrigin(0.5).setAlpha(0.95);
        this.tweens.add({
            targets: gainTxt, y: targetY, alpha: 0, duration: 900, ease: 'Power2',
            onComplete: () => gainTxt.destroy(),
        });

        if (comboCount > 1) {
            const comboFS = Math.floor(Math.min(fs * (1 + comboCount * 0.12), fs * 1.9));
            const comboX = cx + gainTxt.width * 0.5 + 10;
            const comboTxt = this.add.text(comboX, cy, '×' + comboCount, {
                fontFamily: 'Orbitron, Arial', fontSize: comboFS + 'px', fontStyle: 'bold',
                color: '#00e5ff',
                shadow: { offsetX: 0, offsetY: 0, color: '#0088ff', blur: 24, fill: true },
            }).setOrigin(0, 0.5).setAlpha(0.95);
            this.tweens.add({
                targets: comboTxt, y: targetY, alpha: 0, duration: 900, ease: 'Power2',
                onComplete: () => comboTxt.destroy(),
            });
        }
    }

    updateState(newState, lastPieceId) {
        const prevBoard = this.gameState ? this.gameState.board.map(r => [...r]) : null;
        const prevScore = this.gameState ? this.gameState.score : 0;

        if (lastPieceId && prevBoard) {
            this.updateColorGrid(prevBoard, newState.board, lastPieceId);
        } else if (!prevBoard) {
            this.colorGrid = Array.from({ length: BOARD_N }, () => Array(BOARD_N).fill(null));
        }

        this.gameState = newState;
        this.selectedIdx = null;
        this.ghostGfx.clear();

        this.renderBoard();
        this.renderPanel();
        this.scoreLbl.setText('Score: ' + newState.score);
        this.comboLbl.setText('');

        if (prevBoard && lastPieceId) {
            this.animatePlacement(prevBoard, newState.board);
            const cleared = this.findClearedCells(prevBoard, newState.board);
            if (cleared.length > 0) {
                try { this.sound.play('line-clear', { volume: getSFXVolume() }); } catch (e) {}
                this.time.delayedCall(80, () => this.animateClear(cleared));
                this.time.delayedCall(120, () => this.showScorePop(newState.score - prevScore, newState.combo_count));
            }
        }

        if (newState.game_over) {
            this.time.delayedCall(600, () => {
                if (this.ws) this.ws.close();
                this.scene.start('GameOverScene', { score: newState.score });
            });
        }
    }

    async handleBoardClick(cursorRow, cursorCol) {
        if (this.selectedIdx === null || !this.gameState) return;
        const piece = this.gameState.pieces[this.selectedIdx];
        if (!piece) return;

        const { row, col } = this.computeAnchor(piece.grid, cursorRow, cursorCol);
        if (!this.canPlace(piece.grid, row, col)) return;

        const res = await fetch('/api/place', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ piece_id: piece.id, row, col }),
        });
        if (!res.ok) return;
        const state = await res.json();
        this.updateState(state, state.last_piece_id || piece.id);
    }

    async startHumanGame() {
        const res = await fetch('/api/new-game', { method: 'POST' });
        const state = await res.json();
        this.updateState(state, null);
    }

    async resumeHumanGame() {
        const res = await fetch('/api/state');
        const state = await res.json();
        if (!state || state.error) {
            this.startHumanGame();
        } else {
            this.updateState(state, null);
        }
    }

    startAIWatch() {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${proto}//${window.location.host}/ws/ai`);
        this.ws.onopen = () => {
            this.ws.send(JSON.stringify({ speed: this.aiSpeed, agent: this.aiAgent }));
        };
        this.ws.onmessage = (evt) => {
            const state = JSON.parse(evt.data);
            this.updateState(state, state.last_piece_id || null);
        };
    }

    shutdown() {
        if (this.ws) { this.ws.close(); this.ws = null; }
        this.pieceZones.forEach(z => z.destroy());
        this.pieceZones = [];
        this.bgBlocks = [];
    }
}
