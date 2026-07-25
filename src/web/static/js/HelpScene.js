class HelpScene extends Phaser.Scene {
    constructor() {
        super({ key: 'HelpScene' });
    }

    init(data) {
        this.fromScene = (data && data.from) || 'MenuScene';
    }

    create() {
        this.cameras.main.setBackgroundColor('#070714');

        const W = this.scale.width;
        const H = this.scale.height;
        const cx = W / 2;

        this.add.text(cx, H * 0.07, 'How to Play', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(Math.min(H * 0.065, 58)) + 'px',
            fontStyle: 'bold',
            color: '#00d4ff',
            shadow: { offsetX: 0, offsetY: 0, color: '#00aaff', blur: 36, fill: true },
        }).setOrigin(0.5);

        const pad = Math.floor(W * 0.05);
        const divX = cx;
        const colW = Math.floor(W / 2 - pad * 1.4);
        const contentY = H * 0.17;
        const maxContentH = H * 0.74;

        const divGfx = this.add.graphics();
        divGfx.lineStyle(1, 0x0d2244, 1);
        divGfx.lineBetween(divX, contentY, divX, contentY + maxContentH);

        const colTitleFS = Math.floor(Math.min(H * 0.030, 26));
        const colTitleY = contentY;
        const colBodyY = contentY + Math.floor(H * 0.06);

        this.add.text(pad, colTitleY, 'Gameplay', {
            fontFamily: 'Orbitron, Arial',
            fontSize: colTitleFS + 'px',
            fontStyle: 'bold',
            color: '#ffffff',
        }).setOrigin(0, 0);

        this.add.text(divX + pad * 0.8, colTitleY, 'AI Agents', {
            fontFamily: 'Orbitron, Arial',
            fontSize: colTitleFS + 'px',
            fontStyle: 'bold',
            color: '#ffffff',
        }).setOrigin(0, 0);

        const colMaxH = maxContentH - Math.floor(H * 0.06);
        this._scrollAreas = [];
        this.buildColumn(pad, colBodyY, colW, colMaxH, H, this.gameplayLines());
        this.buildColumn(divX + pad * 0.8, colBodyY, colW, colMaxH, H, this.aiLines());

        this.buildBackButton(cx, H);

        this.input.on('wheel', this._onWheel, this);

        this.scale.on('resize', this._onResize, this);
        this.events.once('shutdown', () => this.scale.off('resize', this._onResize, this));
    }

    buildColumn(x, startY, colW, maxH, H, lines) {
        const bodyFS = Math.floor(Math.min(H * 0.020, 16));
        const headFS = Math.floor(Math.min(H * 0.024, 19));
        const lineGap = Math.floor(H * 0.012);
        const headGap = Math.floor(H * 0.022);

        const container = this.add.container(x, startY);
        let y = 0;

        lines.forEach(({ text, type }) => {
            if (type === 'heading') {
                const t = this.add.text(0, y, text, {
                    fontFamily: 'Orbitron, Arial',
                    fontSize: headFS + 'px',
                    fontStyle: 'bold',
                    color: '#00d4ff',
                }).setOrigin(0, 0);
                container.add(t);
                y += t.height + headGap * 0.5;
            } else if (type === 'rule') {
                const g = this.add.graphics();
                g.lineStyle(1, 0x0d2244, 1);
                g.lineBetween(0, y + lineGap, colW, y + lineGap);
                container.add(g);
                y += lineGap * 2.2;
            } else {
                const t = this.add.text(0, y, text, {
                    fontFamily: 'Orbitron, Arial',
                    fontSize: bodyFS + 'px',
                    color: '#7a9cc0',
                    wordWrap: { width: colW },
                }).setOrigin(0, 0);
                container.add(t);
                y += t.height + lineGap;
            }
        });

        const contentH = y;

        const maskGfx = this.make.graphics({}, false);
        maskGfx.fillStyle(0xffffff);
        maskGfx.fillRect(x, startY, colW, maxH);
        container.setMask(maskGfx.createGeometryMask());

        if (contentH > maxH) {
            this.setupScrollbar(container, x, startY, colW, maxH, contentH);
        }
    }

    setupScrollbar(container, x, startY, colW, maxH, contentH) {
        const maxScroll = contentH - maxH;
        const trackX = x + colW + 10;
        const thumbH = Math.max(24, maxH * (maxH / contentH));

        const track = this.add.graphics();
        track.fillStyle(0x0d2244, 0.6);
        track.fillRoundedRect(trackX, startY, 4, maxH, 2);

        const thumb = this.add.graphics();
        const drawThumb = (scrollY) => {
            thumb.clear();
            const t = maxScroll > 0 ? scrollY / maxScroll : 0;
            const thumbY = startY + t * (maxH - thumbH);
            thumb.fillStyle(0x2a6ab0, 0.9);
            thumb.fillRoundedRect(trackX, thumbY, 4, thumbH, 2);
        };
        drawThumb(0);

        const area = {
            container, x, y: startY, w: colW + 20, h: maxH,
            scrollY: 0, maxScroll,
            apply: (scrollY) => {
                area.scrollY = Phaser.Math.Clamp(scrollY, 0, maxScroll);
                container.y = startY - area.scrollY;
                drawThumb(area.scrollY);
            },
        };
        this._scrollAreas.push(area);

        const dragZone = this.add.zone(x, startY, colW, maxH).setOrigin(0, 0).setInteractive();
        dragZone.on('pointerdown', (pointer) => {
            area._dragStartY = pointer.y;
            area._dragStartScroll = area.scrollY;
        });
        this.input.on('pointermove', (pointer) => {
            if (!pointer.isDown || area._dragStartY === undefined) return;
            area.apply(area._dragStartScroll - (pointer.y - area._dragStartY));
        });
        this.input.on('pointerup', () => { area._dragStartY = undefined; });
    }

    _onWheel(pointer, over, dx, dy) {
        const area = (this._scrollAreas || []).find(a =>
            pointer.x >= a.x && pointer.x <= a.x + a.w &&
            pointer.y >= a.y && pointer.y <= a.y + a.h);
        if (area) area.apply(area.scrollY + dy);
    }

    gameplayLines() {
        return [
            { type: 'heading', text: 'Objective' },
            { type: 'body',    text: 'Place pieces on the 8×8 grid to fill complete rows or columns. Clearing them scores points. The game ends when no piece can be placed.' },
            { type: 'rule' },
            { type: 'heading', text: 'Controls' },
            { type: 'body',    text: '① Click a piece in the bottom panel to select it.' },
            { type: 'body',    text: '② Hover over the board — a ghost preview shows placement.' },
            { type: 'body',    text: '③ Click to place. A red ghost means the cell is blocked.' },
            { type: 'rule' },
            { type: 'heading', text: 'Scoring' },
            { type: 'body',    text: 'Each filled cell = 1 pt. Clearing lines scores a bonus: lines × 10 × combo multiplier.' },
            { type: 'body',    text: 'Consecutive clears build a combo. Placing 3 pieces without a clear resets it.' },
            { type: 'body',    text: 'Clearing the entire board awards a +360 bonus.' },
        ];
    }

    aiLines() {
        return [
            { type: 'heading', text: 'Random Agent' },
            { type: 'body',    text: 'Selects both the piece and placement uniformly at random from all valid moves each turn. Makes no attempt to evaluate board state or future consequences. Used as the lower-bound baseline — any agent that cannot outperform random is considered ineffective.' },
            { type: 'rule' },
            { type: 'heading', text: 'Greedy Agent  —  coming soon' },
            { type: 'body',    text: 'Scores every valid (piece, position) combination using a heuristic function and immediately picks the highest-scoring move. The heuristic weighs factors such as cells placed, lines cleared, and board compactness. Efficient and consistent, but vulnerable to local optima — it cannot sacrifice short-term score for a better long-term board state.' },
            { type: 'rule' },
            { type: 'heading', text: 'Beam Search  —  coming soon' },
            { type: 'body',    text: 'Performs a forward tree search, expanding the K most promising board states at each depth level (the beam width). By exploring multiple hypothetical futures simultaneously, it can identify move sequences that sacrifice immediate score for a higher-value position several turns ahead. Stronger than Greedy at the cost of greater computation.' },
            { type: 'rule' },
            { type: 'heading', text: 'DQN Agent  —  coming soon' },
            { type: 'body',    text: 'A Deep Q-Network agent trained via reinforcement learning. A neural network learns to map board states directly to action values (Q-values), guided by rewards accumulated over millions of self-play episodes. Unlike rule-based agents, DQN discovers its own board evaluation strategy through experience rather than hand-crafted heuristics.' },
            { type: 'rule' },
            { type: 'heading', text: 'MCTS Agent  —  coming soon' },
            { type: 'body',    text: 'Monte Carlo Tree Search builds a search tree by repeatedly simulating random games (rollouts) from the current state. Each iteration selects nodes using the UCB1 formula to balance exploration and exploitation. The move with the highest average simulated outcome is chosen. Requires no training and no domain heuristic, relying purely on statistical evidence from simulations.' },
        ];
    }

    buildBackButton(cx, H) {
        const W = Math.floor(H * 0.38);
        const Ht = Math.floor(H * 0.072);
        const R = Math.floor(Ht * 0.22);
        const fs = Math.floor(H * 0.026);
        const y = H * 0.94;

        const gfx = this.add.graphics();
        this.drawBtn(gfx, cx, y, W, Ht, R, false);

        const txt = this.add.text(cx, y, '← Back', {
            fontFamily: 'Orbitron, Arial',
            fontSize: fs + 'px',
            fontStyle: 'bold',
            color: '#c8e8ff',
        }).setOrigin(0.5);

        const zone = this.add.zone(cx - W / 2, y - Ht / 2, W, Ht)
            .setOrigin(0, 0)
            .setInteractive({ useHandCursor: true });

        zone.on('pointerover', () => { this.drawBtn(gfx, cx, y, W, Ht, R, true); txt.setStyle({ color: '#00d4ff' }); });
        zone.on('pointerout',  () => { this.drawBtn(gfx, cx, y, W, Ht, R, false); txt.setStyle({ color: '#c8e8ff' }); });
        zone.on('pointerdown', () => {
            try { this.sound.play('click', { volume: getSFXVolume() }); } catch (e) {}
            this.time.delayedCall(140, () => this.scene.start(this.fromScene));
        });
    }

    drawBtn(gfx, x, y, W, H, R, hover) {
        gfx.clear();
        gfx.fillStyle(hover ? 0x0a2060 : 0x060d28);
        gfx.fillRoundedRect(x - W / 2, y - H / 2, W, H, R);
        gfx.lineStyle(2, hover ? 0x00d4ff : 0x1a4888);
        gfx.strokeRoundedRect(x - W / 2, y - H / 2, W, H, R);
    }

    _onResize() {
        clearTimeout(this._resizeTimer);
        this._resizeTimer = setTimeout(() => this.scene.restart({ from: this.fromScene }), 200);
    }
}
